#!/usr/bin/env bash
# wire_inference.sh — KEPCO OPGW 6-model inference
# anomaly/ + normal/ 하위 폴더 있으면 Precision/Recall/F1/AUROC 자동 계산
# 항상 10-fold mean±std 테이블 출력
#
# Usage:
#   bash wire_inference.sh --test-dir /path/to/dataset
#   bash wire_inference.sh --test-dir /path/to/dataset --fold auto
#   bash wire_inference.sh --test-dir /path/to/images --models vitb_cla_sft2,differnet

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_CKPT_DIR="${SCRIPT_DIR}/checkpoints/wire_final_train"

# ── 기본값 ──────────────────────────────────────────────────────────
FOLD_ARG="9"
TEST_DIR=""
CKPT_DIR="$DEFAULT_CKPT_DIR"
OUTPUT_DIR="${SCRIPT_DIR}/predictions"
METRIC="f1"
MODELS="all"
SPLIT_CSV=""    # 단일 CSV 파일 경로
SPLIT_DIR="${SCRIPT_DIR}/dataset/splits/kfold10_train_val_test_dataset_0622"    # fold_N.csv 들이 있는 디렉토리 (fold별 자동 선택)

usage() {
  cat <<'EOF'

Usage: bash wire_inference.sh --test-dir <path> [options]

필수:
  --test-dir PATH          테스트 이미지 디렉토리
                           (하위에 anomaly/ + normal/ 있으면 메트릭 자동 계산)

옵션:
  --fold N|auto            fold 번호 or 'auto' (기본: 9)
  --metric f1|auroc        auto 기준 지표 (기본: f1)
  --checkpoints-dir PATH   통합 체크포인트 루트
                           (기본: checkpoints/wire_final_train/)
  --split-csv PATH         fold CSV 파일 — test split 이미지만 추론 (공정 평가용)
  --split-dir PATH         fold CSV 디렉토리 — fold_N.csv 자동 선택
  --output-dir PATH        결과 저장 경로 (기본: predictions/)
  --models MODEL,...       실행할 모델 콤마 구분 (기본: all)

모델 (논문 표 순서):
  patchcore           PatchCore               (metrics.json 기반)
  differnet           DifferNet               (predict.py 사용)
  convnextb_linear    ConvNeXt-B (linear)
  vitb_linear         ViT-B (linear)
  convnextb_cla_sft2  Ours (ConvNeXt-B + CLAdapter)
  vitb_cla_sft2       Ours (ViT-B + CLAdapter)   ← 최고 성능, 권장

예시:
  bash wire_inference.sh --test-dir dataset/Dataset_0622
  bash wire_inference.sh --test-dir dataset/Dataset_0622 --fold auto
  bash wire_inference.sh --test-dir /path/to/new_images --models vitb_cla_sft2
  # 공정 평가 (test split만)
  bash wire_inference.sh --test-dir dataset/Dataset_0622 --fold 9 \
    --split-dir dataset/splits/kfold10_train_val_test_dataset_0622

EOF
  exit 1
}

# ── 인자 파싱 ────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --test-dir)        TEST_DIR="$2";   shift 2 ;;
    --fold)            FOLD_ARG="$2";   shift 2 ;;
    --metric)          METRIC="$2";     shift 2 ;;
    --checkpoints-dir) CKPT_DIR="$2";   shift 2 ;;
    --split-csv)       SPLIT_CSV="$2";  shift 2 ;;
    --split-dir)       SPLIT_DIR="$2";  shift 2 ;;
    --output-dir)      OUTPUT_DIR="$2"; shift 2 ;;
    --models)          MODELS="$2";     shift 2 ;;
    -h|--help) usage ;;
    *) echo "[ERROR] 알 수 없는 옵션: $1"; usage ;;
  esac
done


[[ -z "$TEST_DIR" ]]   && { echo "[ERROR] --test-dir 필수"; usage; }
[[ ! -d "$TEST_DIR" ]] && { echo "[ERROR] 경로 없음: $TEST_DIR"; exit 1; }
[[ ! -d "$CKPT_DIR" ]] && { echo "[ERROR] 체크포인트 없음: $CKPT_DIR"; exit 1; }

# fold opts (CLAdapter inference.py 용)
if [[ "$FOLD_ARG" == "auto" ]]; then
  FOLD_OPTS="--auto-best --metric $METRIC"
  FOLD_TAG="auto"
else
  FOLD_OPTS="--fold $FOLD_ARG"
  FOLD_TAG="fold${FOLD_ARG}"
fi

# 모델 목록 (논문 표 순서)
if [[ "$MODELS" == "all" ]]; then
  MODEL_LIST="patchcore differnet convnextb_linear vitb_linear convnextb_cla_sft2 vitb_cla_sft2"
else
  MODEL_LIST="${MODELS//,/ }"
fi

# GT 라벨 가능 여부 (대소문자 무관)
HAS_GT=false
DIR_AN="" DIR_NR=""
for _a in anomaly Anomaly; do
  for _n in normal Normal; do
    if [[ -d "${TEST_DIR}/${_a}" && -d "${TEST_DIR}/${_n}" ]]; then
      HAS_GT=true; DIR_AN="${TEST_DIR}/${_a}"; DIR_NR="${TEST_DIR}/${_n}"; break 2
    fi
  done
done

mkdir -p "$OUTPUT_DIR"
TMP_DIR=$(mktemp -d)
trap "rm -rf $TMP_DIR" EXIT

# 모델 Display Name
dn() {
  case $1 in
    vitb_cla_sft2)      echo "Ours (ViT-B + CLAdapter)" ;;
    convnextb_cla_sft2) echo "Ours (ConvNeXt-B + CLAdapter)" ;;
    vitb_linear)        echo "ViT-B" ;;
    convnextb_linear)   echo "ConvNeXt-B" ;;
    patchcore)          echo "PatchCore" ;;
    differnet)          echo "DifferNet" ;;
    *)                  echo "$1" ;;
  esac
}

TOTAL_MODELS=$(echo $MODEL_LIST | wc -w)
SCRIPT_START=$SECONDS

printf "==================================================\n"
printf " KEPCO OPGW Wire Inference\n"
printf " test-dir  : %s\n" "$TEST_DIR"
printf "==================================================\n"

FAILED=()
MODEL_IDX=0

for MODEL in $MODEL_LIST; do
  MODEL_IDX=$((MODEL_IDX + 1))
  MODEL_START=$SECONDS
  printf "\n[%d/%d] %-32s" "$MODEL_IDX" "$TOTAL_MODELS" "$(dn $MODEL)"

  # patchcore / differnet auto fold 선택
  FOLD_NUM="$FOLD_ARG"
  if [[ "$FOLD_ARG" == "auto" && ( "$MODEL" == "patchcore" || "$MODEL" == "differnet" ) ]]; then
    FOLD_NUM=$(MODEL_ARG="$MODEL" CKPT_ARG="$CKPT_DIR" METRIC_ARG="$METRIC" python3 <<'PYEOF'
import os, json
from pathlib import Path
model  = os.environ["MODEL_ARG"]
ckpt   = Path(os.environ["CKPT_ARG"])
metric = os.environ["METRIC_ARG"]
subdir = "patchcore" if model == "patchcore" else "differnet"
best_fold, best_val = 0, -1.0
for fd in sorted(ckpt.glob("fold_*")):
    mp = fd / subdir / "metrics.json"
    if not mp.exists():
        continue
    m  = json.load(open(mp))
    fn = int(fd.name.split("_")[1])
    if model == "patchcore":
        val = m.get("test",{}).get("best_threshold",{}).get("f1", 0) if metric == "f1" \
              else m.get("test",{}).get("auroc", 0)
    else:
        c   = m.get("test",{}).get("confusion",{})
        tp  = c.get("tp",0); fp = c.get("fp",0); fnn = c.get("fn",0)
        pr  = tp/(tp+fp)  if tp+fp  > 0 else 0
        rc  = tp/(tp+fnn) if tp+fnn > 0 else 0
        val = 2*pr*rc/(pr+rc) if pr+rc > 0 else 0
    if val > best_val:
        best_val, best_fold = val, fn
print(best_fold)
PYEOF
    )
    printf "  (auto→fold%s)" "$FOLD_NUM"
  fi

  # ── split CSV 기반 test 이미지 필터링 ─────────────────────────────
  # --split-csv 또는 --split-dir 지정 시 test split 이미지만 추론
  INFER_ANOMALY="$DIR_AN"
  INFER_NORMAL="$DIR_NR"
  USE_SPLIT=false

  if [[ -n "$SPLIT_CSV" || -n "$SPLIT_DIR" ]]; then
    # 사용할 CSV 결정
    if [[ -n "$SPLIT_CSV" ]]; then
      CUR_CSV="$SPLIT_CSV"
    else
      CUR_CSV="${SPLIT_DIR}/fold_${FOLD_NUM}.csv"
    fi

    if [[ ! -f "$CUR_CSV" ]]; then
      :
    else
      TEST_AN="${TMP_DIR}/${MODEL}_test_an"
      TEST_NR="${TMP_DIR}/${MODEL}_test_nr"
      mkdir -p "$TEST_AN" "$TEST_NR"

      if SPLIT_FILE="$CUR_CSV" DATA_ROOT="$TEST_DIR" \
         OUT_AN="$TEST_AN" OUT_NR="$TEST_NR" python3 <<'PYEOF'
import os, csv
from pathlib import Path

split_file = Path(os.environ["SPLIT_FILE"])
data_root  = Path(os.environ["DATA_ROOT"])
out_an     = Path(os.environ["OUT_AN"])
out_nr     = Path(os.environ["OUT_NR"])

n_an = n_nr = 0
with open(split_file, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        # split 컬럼 (대소문자 무관)
        split_val = (row.get("split") or row.get("Split") or "").strip().lower()
        if split_val != "test":
            continue
        # 이미지 경로 컬럼
        img = (row.get("image_path") or row.get("filepath") or
               row.get("path") or row.get("filename") or "").strip()
        if not img:
            continue
        p = Path(img)
        if not p.is_absolute():
            # 1) data_root 기준, 2) CWD 기준 순으로 시도
            for base in [data_root, data_root.parent, Path.cwd()]:
                candidate = (base / p).resolve()
                if candidate.exists():
                    p = candidate; break
        if not p.exists():
            continue
        parts = str(p).replace("\\", "/").lower()
        if "/anomaly/" in parts or parts.endswith("/anomaly"):
            dst = out_an / p.name
            n_an += 1
        elif "/normal/" in parts or parts.endswith("/normal"):
            dst = out_nr / p.name
            n_nr += 1
        else:
            continue
        if not dst.exists():
            dst.symlink_to(p.resolve())

print(f" test data (normal: {n_nr}장, anomaly: {n_an}장)", end="", flush=True)
PYEOF
      then
        INFER_ANOMALY="$TEST_AN"
        INFER_NORMAL="$TEST_NR"
        USE_SPLIT=true
      else
        echo "  [경고] split 필터링 실패 — 전체 데이터셋으로 대체"
      fi
    fi
  fi

  # ── PatchCore: metrics.json 기반 (standalone inference 미지원) ────
  if [[ "$MODEL" == "patchcore" ]]; then
    PC_JSON=$(find "${CKPT_DIR}/fold_${FOLD_NUM}/patchcore" -name "metrics.json" 2>/dev/null | head -1)
    if [[ ! -f "$PC_JSON" ]]; then
      echo "  [경고] metrics.json 없음: $PC_JSON"
      FAILED+=("$MODEL"); continue
    fi
    if ! IN_JSON="$PC_JSON" OUT_JSON="${OUTPUT_DIR}/patchcore_${FOLD_TAG}.json" python3 <<'PYEOF'
import os, json
m     = json.load(open(os.environ["IN_JSON"]))
block = m.get("test_at_val_best_threshold") or m["test"]["best_threshold"]
c     = block.get("confusion", {})
tn=c.get("tn",0); fp_v=c.get("fp",0); fn_v=c.get("fn",0); tp=c.get("tp",0)
norm_prec=tn/(tn+fn_v+1e-9); norm_rec=tn/(tn+fp_v+1e-9)
an_prec  =tp/(tp+fp_v+1e-9); an_rec  =tp/(tp+fn_v+1e-9)
prec  = (norm_prec+an_prec)/2*100
rec   = (norm_rec +an_rec )/2*100
f1    = 2*prec*rec/(prec+rec) if prec+rec > 0 else 0.0
auroc = m["test"]["auroc"] * 100
json.dump({"precision":prec,"recall":rec,"f1":f1,"auroc":auroc,"source":"metrics_json"},
          open(os.environ["OUT_JSON"],"w"), indent=2)
PYEOF
    then
      FAILED+=("$MODEL")
    fi
    _ELAPSED=$((SECONDS - MODEL_START))
    _M=$(python3 -c "
import json
try:
    r=json.load(open('${OUTPUT_DIR}/patchcore_${FOLD_TAG}.json'))
    print(f\"  Precision={r['precision']:.2f}%  Recall={r['recall']:.2f}%  F1={r['f1']:.2f}%  AUROC={r['auroc']:.2f}%\", end='')
except: pass
" 2>/dev/null)
    printf "  →  완료 (%ds)%s\n" "$_ELAPSED" "$_M"
    continue
  fi

  # ── DifferNet: predict.py 사용 ────────────────────────────────────
  if [[ "$MODEL" == "differnet" ]]; then
    DN_CKPT=$(find "${CKPT_DIR}/fold_${FOLD_NUM}/differnet" -name "best.pt" 2>/dev/null | head -1)
    if [[ ! -f "$DN_CKPT" ]]; then
      echo "  [경고] 체크포인트 없음: $DN_CKPT"
      FAILED+=("$MODEL"); continue
    fi

    if $HAS_GT; then
      if ! python3 -m baselines.attent_differnet.predict \
          --checkpoint "$DN_CKPT" \
          --input "$INFER_ANOMALY" \
          --output-csv "${TMP_DIR}/dn_anomaly.csv" > /dev/null 2>&1; then
        FAILED+=("$MODEL"); continue
      fi
      if ! python3 -m baselines.attent_differnet.predict \
          --checkpoint "$DN_CKPT" \
          --input "$INFER_NORMAL" \
          --output-csv "${TMP_DIR}/dn_normal.csv" > /dev/null 2>&1; then
        FAILED+=("$MODEL"); continue
      fi
      if ! AN_CSV="${TMP_DIR}/dn_anomaly.csv" \
           NR_CSV="${TMP_DIR}/dn_normal.csv" \
           OUT_CSV="${OUTPUT_DIR}/differnet_${FOLD_TAG}.csv" \
           OUT_JSON="${OUTPUT_DIR}/differnet_${FOLD_TAG}.json" python3 <<'PYEOF'
import os, csv, json
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

def read(path, label):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            vals = list(r.values())
            rows.append({"image_path": vals[0], "score": float(vals[1]), "true_label": label})
    return rows

rows   = read(os.environ["AN_CSV"], 1) + read(os.environ["NR_CSV"], 0)
scores = np.array([r["score"] for r in rows])
labels = np.array([r["true_label"] for r in rows])

fpr, tpr, thrs = roc_curve(labels, scores)
thr = thrs[np.argmax(tpr - fpr)]

for r in rows:
    r["pred_label"] = int(r["score"] >= thr)
    r["pred_name"]  = "anomaly" if r["pred_label"] == 1 else "normal"

tp  = sum(1 for r in rows if r["true_label"]==1 and r["pred_label"]==1)
fp  = sum(1 for r in rows if r["true_label"]==0 and r["pred_label"]==1)
fn  = sum(1 for r in rows if r["true_label"]==1 and r["pred_label"]==0)
tn  = sum(1 for r in rows if r["true_label"]==0 and r["pred_label"]==0)
norm_prec=tn/(tn+fn+1e-9); norm_rec=tn/(tn+fp+1e-9)
an_prec  =tp/(tp+fp+1e-9); an_rec  =tp/(tp+fn+1e-9)
prec  = (norm_prec+an_prec)/2*100
rec   = (norm_rec +an_rec )/2*100
f1    = 2*prec*rec/(prec+rec) if prec+rec > 0 else 0.0
auroc = roc_auc_score(labels, scores) * 100
json.dump({"precision":prec,"recall":rec,"f1":f1,"auroc":auroc},
          open(os.environ["OUT_JSON"],"w"), indent=2)
with open(os.environ["OUT_CSV"],"w",newline="",encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["image_path","score","true_label","pred_label","pred_name"])
    w.writeheader(); w.writerows(rows)
PYEOF
      then
        FAILED+=("$MODEL"); continue
      fi
    else
      if ! python3 -m baselines.attent_differnet.predict \
          --checkpoint "$DN_CKPT" \
          --input "$TEST_DIR" \
          --output-csv "${OUTPUT_DIR}/differnet_${FOLD_TAG}.csv" 2>/dev/null; then
        FAILED+=("$MODEL"); continue
      fi
    fi
    _ELAPSED=$((SECONDS - MODEL_START))
    _M=$(python3 -c "
import json
try:
    r=json.load(open('${OUTPUT_DIR}/differnet_${FOLD_TAG}.json'))
    print(f\"  Precision={r['precision']:.2f}%  Recall={r['recall']:.2f}%  F1={r['f1']:.2f}%  AUROC={r['auroc']:.2f}%\", end='')
except: pass
" 2>/dev/null)
    printf "  →  완료 (%ds)%s\n" "$_ELAPSED" "$_M"
    continue
  fi

  # ── CLAdapter 4모델: inference.py 사용 ──────────────────────────
  if $HAS_GT; then
    if ! python3 "${SCRIPT_DIR}/inference/inference.py" \
        --test-dir "$INFER_ANOMALY" --model "$MODEL" \
        $FOLD_OPTS --checkpoints-dir "$CKPT_DIR" \
        --output "${TMP_DIR}/${MODEL}_anomaly.csv" \
        --quiet > /dev/null 2>&1; then
      FAILED+=("$MODEL"); continue
    fi
    if ! python3 "${SCRIPT_DIR}/inference/inference.py" \
        --test-dir "$INFER_NORMAL" --model "$MODEL" \
        $FOLD_OPTS --checkpoints-dir "$CKPT_DIR" \
        --output "${TMP_DIR}/${MODEL}_normal.csv" \
        --quiet > /dev/null 2>&1; then
      FAILED+=("$MODEL"); continue
    fi
    if ! AN_CSV="${TMP_DIR}/${MODEL}_anomaly.csv" \
         NR_CSV="${TMP_DIR}/${MODEL}_normal.csv" \
         OUT_CSV="${OUTPUT_DIR}/${MODEL}_${FOLD_TAG}.csv" python3 <<'PYEOF'
import os, csv
rows = []
for path, lbl in [(os.environ["AN_CSV"], 1), (os.environ["NR_CSV"], 0)]:
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["true_label"] = lbl; rows.append(r)
out = os.environ["OUT_CSV"]
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
PYEOF
    then
      FAILED+=("$MODEL"); continue
    fi
  else
    if ! python3 "${SCRIPT_DIR}/inference/inference.py" \
        --test-dir "$TEST_DIR" --model "$MODEL" \
        $FOLD_OPTS --checkpoints-dir "$CKPT_DIR" \
        --output "${OUTPUT_DIR}/${MODEL}_${FOLD_TAG}.csv" \
        --quiet > /dev/null 2>&1; then
      FAILED+=("$MODEL"); continue
    fi
  fi
  # CLAdapter 완료 stats
  _ELAPSED=$((SECONDS - MODEL_START))
  _OUT="${OUTPUT_DIR}/${MODEL}_${FOLD_TAG}.csv"
  if [[ -f "$_OUT" ]]; then
    _M=$(python3 -c "
import csv
from sklearn.metrics import roc_auc_score
rows=list(csv.DictReader(open('$_OUT')))
if rows:
    yt=[int(r['true_label']) for r in rows]
    yp=[int(r['pred_label']) for r in rows]
    ys=[float(r.get('prob_anomaly',r.get('score',0))) for r in rows]
    tp=sum(1 for a,b in zip(yt,yp) if a==1 and b==1)
    fp=sum(1 for a,b in zip(yt,yp) if a==0 and b==1)
    fn=sum(1 for a,b in zip(yt,yp) if a==1 and b==0)
    pr=tp/(tp+fp)*100 if tp+fp>0 else 0
    rc=tp/(tp+fn)*100 if tp+fn>0 else 0
    f1=2*pr*rc/(pr+rc) if pr+rc>0 else 0
    try:
        auc=roc_auc_score(yt,ys)*100
    except:
        auc=0
    print(f'  Precision={pr:.2f}%  Recall={rc:.2f}%  F1={f1:.2f}%  AUROC={auc:.2f}%', end='')
" 2>/dev/null) || true
    printf "  →  완료 (%ds)%s\n" "$_ELAPSED" "$_M"
  else
    printf "  →  완료 (%ds)\n" "$_ELAPSED"
  fi
done

printf "\n 총 소요: %ds\n" "$((SECONDS - SCRIPT_START))"

# ── 메트릭 테이블 출력 ─────────────────────────────────────────────
MODEL_LIST_PY="$MODEL_LIST" \
FOLD_TAG_PY="$FOLD_TAG" \
OUT_DIR_PY="$OUTPUT_DIR" \
CKPT_DIR_PY="$CKPT_DIR" \
HAS_GT_PY="$HAS_GT" python3 <<'PYEOF'
import os, csv, json, math
from pathlib import Path

model_list = os.environ["MODEL_LIST_PY"].split()
fold_tag   = os.environ["FOLD_TAG_PY"]
out_dir    = Path(os.environ["OUT_DIR_PY"])
ckpt_dir   = Path(os.environ["CKPT_DIR_PY"])
has_gt     = os.environ["HAS_GT_PY"] == "true"

LABELS = {
    "patchcore":          "PatchCore",
    "differnet":          "DifferNet",
    "convnextb_linear":   "ConvNeXt-B",
    "vitb_linear":        "ViT-B",
    "convnextb_cla_sft2": "Ours (ConvNeXt-B + CLAdapter)",
    "vitb_cla_sft2":      "Ours (ViT-B + CLAdapter)",
}
SUBDIRS = {
    "patchcore":          "patchcore",
    "differnet":          "differnet",
    "convnextb_linear":   "linear_convnextb",
    "vitb_linear":        "linear_vitb",
    "convnextb_cla_sft2": "convnextb_cla_sft2",
    "vitb_cla_sft2":      "vitb_cla_sft2",
}

def csv_metrics(path):
    y_true, y_pred, y_prob = [], [], []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            y_true.append(int(r["true_label"]))
            y_pred.append(int(r["pred_label"]))
            pk = "prob_anomaly" if "prob_anomaly" in r else "score"
            y_prob.append(float(r[pk]))
    tp = sum(1 for a,b in zip(y_true,y_pred) if a==1 and b==1)
    fp = sum(1 for a,b in zip(y_true,y_pred) if a==0 and b==1)
    fn = sum(1 for a,b in zip(y_true,y_pred) if a==1 and b==0)
    prec  = tp/(tp+fp)*100 if tp+fp > 0 else 0.0
    rec   = tp/(tp+fn)*100 if tp+fn > 0 else 0.0
    f1    = 2*prec*rec/(prec+rec) if prec+rec > 0 else 0.0
    try:
        from sklearn.metrics import roc_auc_score
        auroc = roc_auc_score(y_true, y_prob) * 100
    except Exception:
        auroc = float("nan")
    return {"precision": prec, "recall": rec, "f1": f1, "auroc": auroc}

def get_val(m, dotkey):
    """flat {"test.f1": 0.99} 와 nested {"test": {"f1": 0.99}} 모두 지원"""
    if dotkey in m:
        return m[dotkey]
    parts = dotkey.split(".", 1)
    if len(parts) == 2:
        return m.get(parts[0], {}).get(parts[1], 0.0)
    return 0.0

def fold_metrics(ckpt_dir, model, fold_n):
    subdir = SUBDIRS[model]
    base = ckpt_dir / f"fold_{fold_n}" / subdir
    if model in ("patchcore", "differnet"):
        candidates = list(base.rglob("metrics.json"))
        if not candidates:
            return None
        mp = candidates[0]
    else:
        mp = base / "metrics.json"
        if not mp.exists():
            return None
    m = json.load(open(mp))
    if model in ("patchcore", "differnet"):
        # 슬라이드 기준: val threshold → test, macro F1 (정상+이상 클래스 평균)
        if model == "patchcore":
            block = m.get("test_at_val_best_threshold") or m.get("test",{}).get("best_threshold",{})
        else:  # differnet
            block = m.get("test_at_val_best_threshold") or {}
            if "confusion" not in block:
                block = {"confusion": m.get("test",{}).get("confusion",{})}
        c  = block.get("confusion",{}) if isinstance(block, dict) else {}
        tn = c.get("tn",0); fp_v = c.get("fp",0); fn_v = c.get("fn",0); tp = c.get("tp",0)
        norm_prec = tn/(tn+fn_v+1e-9)
        norm_rec  = tn/(tn+fp_v+1e-9)
        an_prec   = tp/(tp+fp_v+1e-9)
        an_rec    = tp/(tp+fn_v+1e-9)
        prec  = (norm_prec + an_prec) / 2 * 100
        rec   = (norm_rec  + an_rec ) / 2 * 100
        f1    = 2*prec*rec/(prec+rec) if prec+rec > 0 else 0.0
        auroc = m.get("test",{}).get("auroc", 0) * 100
    else:  # CLAdapter — prec/rec/f1은 이미 % 형태, auroc만 fraction
        prec  = get_val(m, "test.prec")
        rec   = get_val(m, "test.reca")
        f1    = get_val(m, "test.f1")
        auroc = get_val(m, "test.roc") * 100
    return {"precision": prec, "recall": rec, "f1": f1, "auroc": auroc}

# 10-fold mean ± std
kfold = {}
for m in model_list:
    fms = [fold_metrics(ckpt_dir, m, n) for n in range(10)]
    fms = [f for f in fms if f]
    if not fms:
        continue
    # 모든 값이 0이면 데이터 없음으로 간주 (건너뜀)
    all_zero = all(abs(f["f1"]) < 1e-6 for f in fms)
    if all_zero:
        continue
    kfold[m] = {}
    for k in ["precision", "recall", "f1", "auroc"]:
        vals = [f[k] for f in fms]
        mn   = sum(vals) / len(vals)
        std  = (sum((v-mn)**2 for v in vals) / len(vals)) ** 0.5
        kfold[m][k] = {"mean": mn, "std": std, "n": len(vals)}

W = 32

def ms(s, k, single):
    if k not in s:
        return "    N/A      "
    if single:
        return f"{s[k]['mean']:6.2f}     "
    return f"{s[k]['mean']:6.2f}±{s[k]['std']:5.2f}"

# ── 10-fold mean±std 테이블 ───────────────────────────────────────
if kfold:
    n_folds = max(v["f1"]["n"] for v in kfold.values())
    single  = n_folds == 1
    print()
    print("=" * 88)
    if single:
        print(f"  [ 결과 (n={n_folds}) ]")
    else:
        print(f"  [ 10-Fold 평균 ± 표준편차  (n={n_folds}) ]")
    print(f"  {'Model':<{W}} {'Precision':>15} {'Recall':>15} {'F1 Score':>15} {'AUROC':>15}")
    print("-" * 88)
    for m in model_list:
        if m not in kfold:
            continue
        s   = kfold[m]
        lbl = LABELS.get(m, m)
        print(f"  {lbl:<{W}} {ms(s,'precision',single):>15} {ms(s,'recall',single):>15} {ms(s,'f1',single):>15} {ms(s,'auroc',single):>15}")
    print("=" * 88)
    print()

# 저장
summary = {
    "fold_tag": fold_tag,
    "kfold10":  {
        LABELS.get(m,m): {k: {"mean": v["mean"], "std": v["std"]} for k,v in kf.items()}
        for m, kf in kfold.items()
    },
}
(out_dir / f"summary_{fold_tag}.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
txt_lines = [
    f"{LABELS.get(m,m)}: "
    f"P={ms(kfold[m],'precision',single)} "
    f"R={ms(kfold[m],'recall',single)} "
    f"F1={ms(kfold[m],'f1',single)} "
    f"AUC={ms(kfold[m],'auroc',single)}"
    for m in model_list if m in kfold
]
(out_dir / f"summary_{fold_tag}.txt").write_text("\n".join(txt_lines), encoding="utf-8")
print(f"  요약 저장: {out_dir}/summary_{fold_tag}.{{txt,json}}")
PYEOF


echo ""
echo "=================================================="
echo " 완료. 결과: $OUTPUT_DIR/"
ls "$OUTPUT_DIR/" 2>/dev/null || true
[[ ${#FAILED[@]} -gt 0 ]] && echo " 실패 모델: ${FAILED[*]}"
echo "=================================================="
