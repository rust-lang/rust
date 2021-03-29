# Note to people running shellcheck: this file should only be sourced, not executed directly.

set -e

dylib=$(echo "" | rustc --print file-names --crate-type dylib --crate-name rustc_codegen_cranelift -)

if echo "$RUSTC_WRAPPER" | grep sccache; then
echo
echo -e "\x1b[1;93m=== Warning: Unset RUSTC_WRAPPER to prevent interference with sccache ===\x1b[0m"
echo
export RUSTC_WRAPPER=
fi

dir=$(cd "$(dirname "${BASH_SOURCE[0]}")"; pwd)

export RUSTC=$dir"/bin/cg_clif"

export RUSTDOCFLAGS=$linker' -Cpanic=abort -Zpanic-abort-tests '\
'-Zcodegen-backend='$dir'/lib/'$dylib' --sysroot '$dir

# FIXME fix `#[linkage = "extern_weak"]` without this
if [[ "$(uname)" == 'Darwin' ]]; then
   export RUSTFLAGS="$RUSTFLAGS -Clink-arg=-undefined -Clink-arg=dynamic_lookup"
fi

export LD_LIBRARY_PATH="$(rustc --print sysroot)/lib:"$dir"/lib"
export DYLD_LIBRARY_PATH=$LD_LIBRARY_PATH
