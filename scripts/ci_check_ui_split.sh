#!/bin/bash
set -e

echo "Checking UI Architecture Invariants..."

# 1. Bloom Forbidden Dependencies
FORBIDDEN=("usvg" "tiny-skia" "fontdue" "resvg")
EXIT_CODE=0

for dep in "${FORBIDDEN[@]}"; do
    if grep -q "$dep" userspace/bloom/Cargo.toml; then
        echo "FAIL: Bloom depends on $dep"
        EXIT_CODE=1
    fi
done

# 2. Bloom Forbidden Modules
if [ -d "userspace/bloom/src/svg" ]; then
    echo "FAIL: Bloom still has src/svg directory"
    EXIT_CODE=1
fi

# 3. Only Blossom writes UI_PRESENT_EPOCH
# Heuristic: look for prop_set calls with UI_PRESENT_EPOCH
# exclude target directory
FOUND_WRITES=$(grep -r "prop_set.*keys::UI_PRESENT_EPOCH" userspace/ | grep -v "userspace/blossom/" | grep -v "target/" || true)

if [ ! -z "$FOUND_WRITES" ]; then
    echo "FAIL: Forbidden writes to UI_PRESENT_EPOCH found outside Blossom:"
    echo "$FOUND_WRITES"
    EXIT_CODE=1
fi

if [ $EXIT_CODE -eq 0 ]; then
    echo "UI Architecture Check PASSED."
else
    echo "UI Architecture Check FAILED."
    exit 1
fi
