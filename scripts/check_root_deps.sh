#!/usr/bin/env bash
# scripts/check_root_deps.sh
#
# janix ACT I: Sever the Graph Spine — regression guard.
#
# This script verifies that no new `crate::root` or `use crate::root`
# dependencies are introduced outside the explicitly allowed set of files.
#
# Run from the repository root:
#   bash scripts/check_root_deps.sh
#
# Exit codes:
#   0 — no violations
#   1 — one or more violations found
#
# To add a permanent exemption, append the file path (relative to
# kernel/src/) to the ALLOWED array below.

set -euo pipefail

KERNEL_SRC="kernel/src"

# Files that are explicitly permitted to use crate::root.
# Keep this list minimal and well-justified.
ALLOWED=(
    # Root service implementation — obviously allowed.
    "root/"
    # Background graph-observer tasks (janix ACT I quarantined layer).
    "task/graph.rs"
    "task/graph_queue.rs"
    "task/graphify.rs"
    "task/flusher.rs"
    # Root syscall handlers.
    "syscall/handlers/root_handlers.rs"
    # Watch-based waiting (syscall surface for userspace graph watches).
    "syscall/handlers/wait.rs"
    # Session layer.
    "petals_session/"
    # MSI/PCI — uses root for device registration.
    "irq/msi.rs"
    # Top-level boot sequence — initialises Root and calls init_graph_workers.
    "lib.rs"
    # Structured logging (uses Root for LogEvent).
    "logging.rs"
    # Syscall dispatch helpers.
    "syscall/handlers/mod.rs"
    # Structured logging helper.
    "syscall/handlers/logging.rs"
)

VIOLATIONS=0

while IFS= read -r -d '' file; do
    # Compute path relative to kernel/src/
    rel="${file#$KERNEL_SRC/}"

    # Check if this file is in the allowed list.
    allowed=0
    for pattern in "${ALLOWED[@]}"; do
        if [[ "$rel" == $pattern* ]]; then
            allowed=1
            break
        fi
    done

    if [[ $allowed -eq 1 ]]; then
        continue
    fi

    # Search for crate::root references (imports or inline paths).
    if grep -qE 'crate::root[[:space:]:;,]' "$file" 2>/dev/null; then
        echo "VIOLATION: $file uses crate::root but is not in the allowed list."
        grep -nE 'crate::root[[:space:]:;,]' "$file" | head -5
        echo
        VIOLATIONS=$((VIOLATIONS + 1))
    fi
done < <(find "$KERNEL_SRC" -name '*.rs' -print0)

if [[ $VIOLATIONS -gt 0 ]]; then
    echo "ERROR: $VIOLATIONS file(s) introduce new Root dependencies."
    echo "See kernel/src/root/mod.rs for the allowed caller list."
    echo "To add a permanent exemption, edit scripts/check_root_deps.sh."
    exit 1
fi

echo "OK: No unexpected Root dependencies found."
exit 0
