#!/usr/bin/env bash
set -e

cd $(dirname "$0")/../

source ./scripts/setup_rust_fork.sh

echo "[TEST] Test suite of rustc"
pushd rust

command -v rg >/dev/null 2>&1 || cargo install ripgrep

rm -r tests/ui/{extern/,unsized-locals/,lto/,linkage*} || true
for test in $(rg --files-with-matches "lto|// needs-asm-support|// needs-unwind" tests/{ui,incremental}); do
  rm $test
done

for test in $(rg -i --files-with-matches "//(\[\w+\])?~[^\|]*\s*ERR|// error-pattern:|// build-fail|// run-fail|-Cllvm-args" tests/ui); do
  rm $test
done

git checkout -- tests/ui/issues/auxiliary/issue-3136-a.rs # contains //~ERROR, but shouldn't be removed
git checkout -- tests/ui/proc-macro/pretty-print-hack/

# missing features
# ================

# requires stack unwinding
rm tests/incremental/change_crate_dep_kind.rs
rm tests/incremental/issue-80691-bad-eval-cache.rs # -Cpanic=abort causes abort instead of exit(101)

# requires compiling with -Cpanic=unwind
rm -r tests/ui/macros/rfc-2011-nicer-assert-messages/
rm -r tests/run-make/test-benches

# vendor intrinsics
rm tests/ui/sse2.rs # cpuid not supported, so sse2 not detected
rm tests/ui/intrinsics/const-eval-select-x86_64.rs # requires x86_64 vendor intrinsics
rm tests/ui/simd/array-type.rs # "Index argument for `simd_insert` is not a constant"
rm tests/ui/simd/intrinsic/generic-bitmask-pass.rs # simd_bitmask unimplemented
rm tests/ui/simd/intrinsic/generic-as.rs # simd_as unimplemented
rm tests/ui/simd/intrinsic/generic-arithmetic-saturating-pass.rs # simd_saturating_add unimplemented
rm tests/ui/simd/intrinsic/float-math-pass.rs # simd_fcos unimplemented
rm tests/ui/simd/intrinsic/generic-gather-pass.rs # simd_gather unimplemented
rm tests/ui/simd/intrinsic/generic-select-pass.rs # simd_select_bitmask unimplemented
rm tests/ui/simd/issue-85915-simd-ptrs.rs # simd_gather unimplemented
rm tests/ui/simd/issue-89193.rs # simd_gather unimplemented
rm tests/ui/simd/simd-bitmask.rs # simd_bitmask unimplemented

# exotic linkages
rm tests/ui/issues/issue-33992.rs # unsupported linkages
rm tests/incremental/hashes/function_interfaces.rs # same
rm tests/incremental/hashes/statics.rs # same

# variadic arguments
rm tests/ui/abi/mir/mir_codegen_calls_variadic.rs # requires float varargs
rm tests/ui/abi/variadic-ffi.rs # requires callee side vararg support

# unsized locals
rm -r tests/run-pass-valgrind/unsized-locals

# misc unimplemented things
rm tests/ui/intrinsics/intrinsic-nearby.rs # unimplemented nearbyintf32 and nearbyintf64 intrinsics
rm tests/ui/target-feature/missing-plusminus.rs # error not implemented
rm tests/ui/fn/dyn-fn-alignment.rs # wants a 256 byte alignment
rm -r tests/run-make/emit-named-files # requires full --emit support
rm tests/ui/abi/stack-probes.rs # stack probes not yet implemented
rm tests/ui/simd/intrinsic/ptr-cast.rs # simd_expose_addr intrinsic unimplemented
rm -r tests/run-make/repr128-dwarf # debuginfo test
rm tests/codegen-units/item-collection/asm-sym.rs # requires support for sym in asm!()

# optimization tests
# ==================
rm tests/ui/codegen/issue-28950.rs # depends on stack size optimizations
rm tests/ui/codegen/init-large-type.rs # same
rm tests/ui/issues/issue-40883.rs # same
rm -r tests/run-make/fmt-write-bloat/ # tests an optimization

# backend specific tests
# ======================
rm tests/incremental/thinlto/cgu_invalidated_when_import_{added,removed}.rs # requires LLVM
rm tests/ui/abi/stack-protector.rs # requires stack protector support

# giving different but possibly correct results
# =============================================
rm tests/ui/mir/mir_misc_casts.rs # depends on deduplication of constants
rm tests/ui/mir/mir_raw_fat_ptr.rs # same
rm tests/ui/consts/issue-33537.rs # same
rm tests/ui/layout/valid_range_oob.rs # different ICE message

# doesn't work due to the way the rustc test suite is invoked.
# should work when using ./x.py test the way it is intended
# ============================================================
rm -r tests/run-make/emit-shared-files # requires the rustdoc executable in dist/bin/
rm -r tests/run-make/unstable-flag-required # same
rm -r tests/run-make/rustdoc-* # same
rm -r tests/run-make/issue-88756-default-output # same
rm -r tests/run-make/remap-path-prefix-dwarf # requires llvm-dwarfdump
rm -r tests/ui/consts/missing_span_in_backtrace.rs # expects sysroot source to be elsewhere

# genuine bugs
# ============
rm tests/incremental/spike-neg1.rs # errors out for some reason
rm tests/incremental/spike-neg2.rs # same
rm tests/ui/issues/issue-74564-if-expr-stack-overflow.rs # gives a stackoverflow before the backend runs
rm tests/ui/mir/ssa-analysis-regression-50041.rs # produces ICE
rm tests/ui/type-alias-impl-trait/assoc-projection-ice.rs # produces ICE

rm tests/ui/simd/intrinsic/generic-reduction-pass.rs # simd_reduce_add_unordered doesn't accept an accumulator for integer vectors

rm tests/ui/runtime/out-of-stack.rs # SIGSEGV instead of SIGABRT for some reason (#1301)

# bugs in the test suite
# ======================
rm tests/ui/backtrace.rs # TODO warning
rm tests/ui/simple_global_asm.rs # TODO add needs-asm-support
rm tests/ui/test-attrs/test-type.rs # TODO panic message on stderr. correct stdout
# not sure if this is actually a bug in the test suite, but the symbol list shows the function without leading _ for some reason
rm -r tests/run-make/native-link-modifier-bundle
rm tests/ui/process/nofile-limit.rs # TODO some AArch64 linking issue
rm tests/ui/dyn-star/dispatch-on-pin-mut.rs # TODO failed assertion in vtable::get_ptr_and_method_ref

rm tests/ui/stdio-is-blocking.rs # really slow with unoptimized libstd

echo "[TEST] rustc test suite"
RUST_TEST_NOCAPTURE=1 COMPILETEST_FORCE_STAGE0=1 ./x.py test --stage 0 tests/{codegen-units,run-make,run-pass-valgrind,ui,incremental}
popd
