#!/usr/bin/env bash
# Demo script: starts the BCI signal processor in synthetic mode
# and polls /state to show what the OpenClaw LLM would see.
#
# Usage: ./scripts/demo.sh
# Press Ctrl+C to stop.

set -e

PORT="${BCI_PORT:-7680}"
BASE_URL="http://127.0.0.1:${PORT}"

echo "=== BCI-OpenClaw Integration Demo ==="
echo ""
echo "Starting signal processor (synthetic mode) on port ${PORT}..."

# Start server in background
cd "$(dirname "$0")/../signal-processor"
python -m src --synthetic --port "${PORT}" &
SERVER_PID=$!

cleanup() {
    echo ""
    echo "Shutting down..."
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    echo "Done."
}
trap cleanup EXIT INT TERM

# Wait for server to be ready
echo "Waiting for server..."
for i in $(seq 1 30); do
    if curl -sf "${BASE_URL}/health" > /dev/null 2>&1; then
        echo "Server is ready!"
        echo ""
        break
    fi
    sleep 0.5
done

# Poll and display
echo "Polling brain state every 2 seconds..."
echo "This is what the OpenClaw LLM sees via before_prompt_build:"
echo "============================================================"
echo ""

while true; do
    RESPONSE=$(curl -sf "${BASE_URL}/state" 2>/dev/null || echo '{"available":false}')
    AVAILABLE=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('available', False))" 2>/dev/null)

    if [ "$AVAILABLE" = "True" ]; then
        SUMMARY=$(echo "$RESPONSE" | python3 -c "
import sys, json
data = json.load(sys.stdin)
bci = data.get('bci_state', {})
print(bci.get('natural_language_summary', 'N/A'))
" 2>/dev/null)
        STATE=$(echo "$RESPONSE" | python3 -c "
import sys, json
data = json.load(sys.stdin)
bci = data.get('bci_state', {})
state = bci.get('state', {})
scores = bci.get('scores', {})
bp = bci.get('band_powers', {})
sq = bci.get('signal_quality', 0)
print(f'  State:     {state.get(\"primary\", \"?\"):>12} (confidence: {state.get(\"confidence\", 0):.2f})')
print(f'  Attention:     {scores.get(\"attention\", 0):.2f}')
print(f'  Relaxation:    {scores.get(\"relaxation\", 0):.2f}')
print(f'  Cog. Load:     {scores.get(\"cognitive_load\", 0):.2f}')
print(f'  Signal Quality:{sq:.2f}')
print(f'  Band Powers:   d={bp.get(\"delta\",0):.1f} t={bp.get(\"theta\",0):.1f} a={bp.get(\"alpha\",0):.1f} b={bp.get(\"beta\",0):.1f} g={bp.get(\"gamma\",0):.1f}')
" 2>/dev/null)

        echo "[$(date +%H:%M:%S)] LLM Context: ${SUMMARY}"
        echo "$STATE"
        echo ""
    else
        echo "[$(date +%H:%M:%S)] Waiting for BCI data..."
    fi

    sleep 2
done
