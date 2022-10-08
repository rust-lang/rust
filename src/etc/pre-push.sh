#!/bin/sh
# rusty-hook
# version 0.8.4

hookName=$(basename "$0")
gitParams="$*"

if ! command -v rusty-hook >/dev/null 2>&1; then
  if [ -z "${RUSTY_HOOK_SKIP_AUTO_INSTALL}" ]; then
    echo "Finalizing rusty-hook configuration..."
    echo "This may take a few seconds..."
    cargo install rusty-hook >/dev/null 2>&1
  else
    echo "rusty-hook is not installed, and auto install is disabled"
    echo "skipping $hookName hook"
    echo "You can reinstall it using 'cargo install rusty-hook' or delete this hook"
    exit 0
  fi
fi

# echo "rusty-hook version: $(rusty-hook --version)"
# echo "hook file version: 0.8.4"
rusty-hook run --hook "$hookName" "$gitParams"