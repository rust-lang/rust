#!/usr/bin/env bash
set -e

cd $(dirname "$0")/../

source ./scripts/setup_rust_fork.sh

echo "[TEST] Test suite of rustc"
pushd rust

command -v rg >/dev/null 2>&1 || cargo install ripgrep

rm -r tests/ui/{unsized-locals/,lto/,linkage*} || true
for test in $(rg --files-with-matches "lto|// needs-asm-support" tests/{codegen-units,ui,incremental}); do
  rm $test
done

for test in tests/run-make/**/Makefile; do
  if rg "# needs-asm-support" $test >/dev/null; then
    rm -r $(dirname $test)
  fi
done

for test in $(rg -i --files-with-matches "//(\[\w+\])?~[^\|]*\s*ERR|// error-pattern:|// build-fail|// run-fail|-Cllvm-args" tests/ui); do
  rm $test
done

git checkout -- tests/ui/issues/auxiliary/issue-3136-a.rs # contains //~ERROR, but shouldn't be removed
git checkout -- tests/ui/proc-macro/pretty-print-hack/
rm tests/ui/parser/unclosed-delimiter-in-dep.rs # submodule contains //~ERROR

# missing features
# ================

# requires stack unwinding
# FIXME add needs-unwind to this test
rm -r tests/run-make/libtest-junit

# extra warning about -Cpanic=abort for proc macros
rm tests/ui/proc-macro/crt-static.rs
rm tests/ui/proc-macro/proc-macro-deprecated-attr.rs
rm tests/ui/proc-macro/quote-debug.rs
rm tests/ui/proc-macro/no-missing-docs.rs
rm tests/ui/rust-2018/proc-macro-crate-in-paths.rs
rm tests/ui/proc-macro/allowed-signatures.rs

# vendor intrinsics
rm tests/ui/sse2.rs # cpuid not supported, so sse2 not detected
rm tests/ui/simd/array-type.rs # "Index argument for `simd_insert` is not a constant"

# exotic linkages
rm tests/ui/issues/issue-33992.rs # unsupported linkages
rm tests/incremental/hashes/function_interfaces.rs # same
rm tests/incremental/hashes/statics.rs # same

# variadic arguments
rm tests/ui/abi/mir/mir_codegen_calls_variadic.rs # requires float varargs
rm tests/ui/abi/variadic-ffi.rs # requires callee side vararg support
rm -r tests/run-make/c-link-to-rust-va-list-fn # requires callee side vararg support

# unsized locals
rm -r tests/run-pass-valgrind/unsized-locals

# misc unimplemented things
rm tests/ui/intrinsics/intrinsic-nearby.rs # unimplemented nearbyintf32 and nearbyintf64 intrinsics
rm tests/ui/target-feature/missing-plusminus.rs # error not implemented
rm tests/ui/fn/dyn-fn-alignment.rs # wants a 256 byte alignment
rm -r tests/run-make/emit-named-files # requires full --emit support
rm -r tests/run-make/repr128-dwarf # debuginfo test
rm -r tests/run-make/split-debuginfo # same
rm -r tests/run-make/symbols-include-type-name # --emit=asm not supported
rm -r tests/run-make/target-specs # i686 not supported by Cranelift
rm -r tests/run-make/mismatching-target-triples # same
rm -r tests/run-make/use-extern-for-plugins # same

# requires LTO
rm -r tests/run-make/cdylib
rm -r tests/run-make/issue-14500
rm -r tests/run-make/issue-64153
rm -r tests/run-make/codegen-options-parsing
rm -r tests/run-make/lto-*
rm -r tests/run-make/reproducible-build-2
rm -r tests/run-make/issue-109934-lto-debuginfo

# optimization tests
# ==================
rm tests/ui/codegen/issue-28950.rs # depends on stack size optimizations
rm tests/ui/codegen/init-large-type.rs # same
rm tests/ui/issues/issue-40883.rs # same
rm -r tests/run-make/fmt-write-bloat/ # tests an optimization

# backend specific tests
# ======================
rm tests/incremental/thinlto/cgu_invalidated_when_import_{added,removed}.rs # requires LLVM
rm -r tests/run-make/cross-lang-lto # same
rm -r tests/run-make/issue-7349 # same
rm -r tests/run-make/sepcomp-inlining # same
rm -r tests/run-make/sepcomp-separate # same
rm -r tests/run-make/sepcomp-cci-copies # same
rm -r tests/run-make/volatile-intrinsics # same
rm tests/ui/abi/stack-protector.rs # requires stack protector support
rm -r tests/run-make/emit-stack-sizes # requires support for -Z emit-stack-sizes

# giving different but possibly correct results
# =============================================
rm tests/ui/mir/mir_misc_casts.rs # depends on deduplication of constants
rm tests/ui/mir/mir_raw_fat_ptr.rs # same
rm tests/ui/consts/issue-33537.rs # same
rm tests/ui/layout/valid_range_oob.rs # different ICE message

rm tests/ui/consts/issue-miri-1910.rs # different error message
rm tests/ui/consts/offset_ub.rs # same
rm tests/ui/consts/const-eval/ub-slice-get-unchecked.rs # same
rm tests/ui/intrinsics/panic-uninitialized-zeroed.rs # same
rm tests/ui/lint/lint-const-item-mutation.rs # same
rm tests/ui/pattern/usefulness/doc-hidden-non-exhaustive.rs # same
rm tests/ui/suggestions/derive-trait-for-method-call.rs # same
rm tests/ui/typeck/issue-46112.rs # same
rm tests/ui/consts/const_cmp_type_id.rs # same
rm tests/ui/consts/issue-73976-monomorphic.rs # same

# rustdoc-clif passes extra args, suppressing the help message when no args are passed
rm -r tests/run-make/issue-88756-default-output

# doesn't work due to the way the rustc test suite is invoked.
# should work when using ./x.py test the way it is intended
# ============================================================
rm -r tests/run-make/remap-path-prefix-dwarf # requires llvm-dwarfdump
rm -r tests/ui/consts/missing_span_in_backtrace.rs # expects sysroot source to be elsewhere

# genuine bugs
# ============
rm tests/incremental/spike-neg1.rs # errors out for some reason
rm tests/incremental/spike-neg2.rs # same

rm tests/ui/simd/simd-bitmask.rs # simd_bitmask doesn't implement [u*; N] return type

rm -r tests/run-make/issue-51671 # wrong filename given in case of --emit=obj
rm -r tests/run-make/issue-30063 # same
rm -r tests/run-make/multiple-emits # same
rm -r tests/run-make/output-type-permutations # same
rm -r tests/run-make/used # same
rm -r tests/run-make/no-alloc-shim
rm -r tests/run-make/emit-to-stdout

# bugs in the test suite
# ======================
rm tests/ui/backtrace.rs # TODO warning
rm tests/ui/process/nofile-limit.rs # TODO some AArch64 linking issue

rm tests/ui/stdio-is-blocking.rs # really slow with unoptimized libstd

cp ../dist/bin/rustdoc-clif ../dist/bin/rustdoc # some tests expect bin/rustdoc to exist

# prevent $(RUSTDOC) from picking up the sysroot built by x.py. It conflicts with the one used by
# rustdoc-clif
cat <<EOF | git apply -
diff --git a/tests/run-make/tools.mk b/tests/run-make/tools.mk
index ea06b620c4c..b969d0009c6 100644
--- a/tests/run-make/tools.mk
+++ b/tests/run-make/tools.mk
@@ -9,7 +9,7 @@ RUSTC_ORIGINAL := \$(RUSTC)
 BARE_RUSTC := \$(HOST_RPATH_ENV) '\$(RUSTC)'
 BARE_RUSTDOC := \$(HOST_RPATH_ENV) '\$(RUSTDOC)'
 RUSTC := \$(BARE_RUSTC) --out-dir \$(TMPDIR) -L \$(TMPDIR) \$(RUSTFLAGS)
-RUSTDOC := \$(BARE_RUSTDOC) -L \$(TARGET_RPATH_DIR)
+RUSTDOC := \$(BARE_RUSTDOC)
 ifdef RUSTC_LINKER
 RUSTC := \$(RUSTC) -Clinker='\$(RUSTC_LINKER)'
 RUSTDOC := \$(RUSTDOC) -Clinker='\$(RUSTC_LINKER)'
EOF

echo "[TEST] rustc test suite"
COMPILETEST_FORCE_STAGE0=1 ./x.py test --stage 0 --test-args=--nocapture tests/{codegen-units,run-make,run-pass-valgrind,ui,incremental}
popd
