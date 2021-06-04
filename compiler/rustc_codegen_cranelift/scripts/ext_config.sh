# Note to people running shellcheck: this file should only be sourced, not executed directly.

# Various env vars that should only be set for the build system but not for cargo.sh

set -e

export CG_CLIF_DISPLAY_CG_TIME=1
export CG_CLIF_DISABLE_INCR_CACHE=1

export HOST_TRIPLE=$(rustc -vV | grep host | cut -d: -f2 | tr -d " ")
export TARGET_TRIPLE=${TARGET_TRIPLE:-$HOST_TRIPLE}

export RUN_WRAPPER=''
export JIT_SUPPORTED=1
if [[ "$HOST_TRIPLE" != "$TARGET_TRIPLE" ]]; then
   export JIT_SUPPORTED=0
   if [[ "$TARGET_TRIPLE" == "aarch64-unknown-linux-gnu" ]]; then
      # We are cross-compiling for aarch64. Use the correct linker and run tests in qemu.
      export RUSTFLAGS='-Clinker=aarch64-linux-gnu-gcc '$RUSTFLAGS
      export RUN_WRAPPER='qemu-aarch64 -L /usr/aarch64-linux-gnu'
   elif [[ "$TARGET_TRIPLE" == "x86_64-pc-windows-gnu" ]]; then
      # We are cross-compiling for Windows. Run tests in wine.
      export RUN_WRAPPER='wine'
   else
      echo "Unknown non-native platform"
   fi
fi
