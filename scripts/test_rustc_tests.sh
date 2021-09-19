#!/bin/bash
set -e

cd $(dirname "$0")/../

source ./scripts/setup_rust_fork.sh

echo "[TEST] Test suite of rustc"
pushd rust

cargo install ripgrep

rm -r src/test/ui/{extern/,panics/,unsized-locals/,thinlto/,simd*,*lto*.rs,linkage*,unwind-*.rs} || true
for test in $(rg --files-with-matches "asm!|catch_unwind|should_panic|lto|// needs-asm-support" src/test/ui); do
  rm $test
done

for test in $(rg -i --files-with-matches "//(\[\w+\])?~|// error-pattern:|// build-fail|// run-fail|-Cllvm-args" src/test/ui); do
  rm $test
done

git checkout -- src/test/ui/issues/auxiliary/issue-3136-a.rs # contains //~ERROR, but shouldn't be removed

# these all depend on unwinding support
rm src/test/ui/backtrace.rs
rm src/test/ui/array-slice-vec/box-of-array-of-drop-*.rs
rm src/test/ui/array-slice-vec/slice-panic-*.rs
rm src/test/ui/array-slice-vec/nested-vec-3.rs
rm src/test/ui/cleanup-rvalue-temp-during-incomplete-alloc.rs
rm src/test/ui/issues/issue-26655.rs
rm src/test/ui/issues/issue-29485.rs
rm src/test/ui/issues/issue-30018-panic.rs
rm src/test/ui/multi-panic.rs
rm src/test/ui/sepcomp/sepcomp-unwind.rs
rm src/test/ui/structs-enums/unit-like-struct-drop-run.rs
rm src/test/ui/terminate-in-initializer.rs
rm src/test/ui/threads-sendsync/task-stderr.rs
rm src/test/ui/numbers-arithmetic/int-abs-overflow.rs
rm src/test/ui/drop/drop-trait-enum.rs
rm src/test/ui/numbers-arithmetic/issue-8460.rs
rm src/test/ui/rt-explody-panic-payloads.rs
rm src/test/incremental/change_crate_dep_kind.rs

rm src/test/ui/issues/issue-28950.rs # depends on stack size optimizations
rm src/test/ui/init-large-type.rs # same
rm src/test/ui/sse2.rs # cpuid not supported, so sse2 not detected
rm src/test/ui/issues/issue-33992.rs # unsupported linkages
rm src/test/ui/issues/issue-51947.rs # same
rm src/test/ui/numbers-arithmetic/saturating-float-casts.rs # intrinsic gives different but valid result
rm src/test/ui/mir/mir_misc_casts.rs # depends on deduplication of constants
rm src/test/ui/mir/mir_raw_fat_ptr.rs # same
rm src/test/ui/consts/issue-33537.rs # same
rm src/test/ui/async-await/async-fn-size-moved-locals.rs # -Cpanic=abort shrinks some generator by one byte
rm src/test/ui/async-await/async-fn-size-uninit-locals.rs # same
rm src/test/ui/generator/size-moved-locals.rs # same
rm src/test/ui/fn/dyn-fn-alignment.rs # wants a 256 byte alignment
rm src/test/ui/test-attrs/test-fn-signature-verification-for-explicit-return-type.rs # "Cannot run dynamic test fn out-of-process"
rm src/test/ui/intrinsics/intrinsic-nearby.rs # unimplemented nearbyintf32 and nearbyintf64 intrinsics

rm src/test/incremental/hashes/inline_asm.rs # inline asm
rm src/test/incremental/issue-72386.rs # same
rm src/test/incremental/issue-49482.rs # same
rm src/test/incremental/issue-54059.rs # same
rm src/test/incremental/lto.rs # requires lto

rm -r src/test/run-make/emit-shared-files # requires the rustdoc executable in build/bin/
rm -r src/test/run-make/unstable-flag-required # same
rm -r src/test/run-make/emit-named-files # requires full --emit support

rm src/test/pretty/asm.rs # inline asm
rm src/test/pretty/raw-str-nonexpr.rs # same

rm -r src/test/run-pass-valgrind/unsized-locals

rm src/test/ui/json-bom-plus-crlf-multifile.rs # differing warning
rm src/test/ui/json-bom-plus-crlf.rs # same
rm src/test/ui/match/issue-82392.rs # differing error
rm src/test/ui/type-alias-impl-trait/cross_crate_ice*.rs # requires removed aux dep

rm src/test/ui/allocator/no_std-alloc-error-handler-default.rs # missing rust_oom definition
rm src/test/ui/cfg/cfg-panic.rs
rm -r src/test/ui/hygiene/

rm -r src/test/ui/polymorphization/ # polymorphization not yet supported
rm src/test/codegen-units/polymorphization/unused_type_parameters.rs # same

rm -r src/test/run-make/fmt-write-bloat/ # tests an optimization
rm src/test/ui/abi/mir/mir_codegen_calls_variadic.rs # requires float varargs
rm src/test/ui/abi/variadic-ffi.rs # requires callee side vararg support

echo "[TEST] rustc test suite"
RUST_TEST_NOCAPTURE=1 COMPILETEST_FORCE_STAGE0=1 ./x.py test --stage 0 src/test/{codegen-units,run-make,run-pass-valgrind,ui}
popd
