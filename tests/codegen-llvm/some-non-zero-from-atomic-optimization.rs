//! Regression test for <https://github.com/rust-lang/rust/issues/60044>.

// We want to check that `unreachable!()` is optimized away.
//@ compile-flags: -O

// Don't de-duplicate `some_non_zero_from_atomic_get2()` since we want its LLVM IR.
//@ compile-flags: -Zmerge-functions=disabled

// So we don't have to worry about usize.
//@ only-64bit

#![crate_type = "lib"]

use std::num::NonZeroUsize;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::Relaxed;

pub static X: AtomicUsize = AtomicUsize::new(1);

/// We don't need to check the LLVM IR of this function, but we expect its LLVM
/// IR to be identical to `some_non_zero_from_atomic_get2()`.
#[no_mangle]
pub unsafe fn some_non_zero_from_atomic_get() -> Option<NonZeroUsize> {
    let x = X.load(Relaxed);
    Some(NonZeroUsize::new_unchecked(x))
}

/// We want to test that the `unreachable!()` branch is optimized out.
///
/// When that does not happen, the LLVM IR will look like this:
///
/// ```sh
/// rustc +nightly-2024-02-08 --emit=llvm-ir -O -Zmerge-functions=disabled \
///     tests/codegen-llvm/some-non-zero-from-atomic-optimization.rs && \
/// grep -B 1 -A 13 '@some_non_zero_from_atomic_get2()' some-non-zero-from-atomic-optimization.ll
/// ```
/// ```llvm
/// ; Function Attrs: nonlazybind uwtable
/// define noundef i64 @some_non_zero_from_atomic_get2() unnamed_addr #1 {
/// start:
///   %0 = load atomic i64, ptr @_ZN38some_non_zero_from_atomic_optimization1X17h monotonic, align 8
///   %1 = icmp eq i64 %0, 0
///   br i1 %1, label %bb2, label %bb3
///
/// bb2:                                              ; preds = %start
/// ; call core::panicking::panic
///   tail call void @_ZN4core9panicking5panic17h0cc48E(..., ..., ... ) #3
///   unreachable
///
/// bb3:                                              ; preds = %start
///   ret i64 %0
/// }
/// ```
///
/// When it _is_ optimized out, the LLVM IR will look like this:
///
/// ```sh
/// rustc +nightly-2024-02-09 --emit=llvm-ir -O -Zmerge-functions=disabled \
///     tests/codegen-llvm/some-non-zero-from-atomic-optimization.rs && \
/// grep -B 1 -A 6 '@some_non_zero_from_atomic_get2()' some-non-zero-from-atomic-optimization.ll
/// ```
/// ```llvm
/// ; Function Attrs: mustprogress nofree nounwind nonlazybind willreturn memory(...) uwtable
/// define noundef i64 @some_non_zero_from_atomic_get2() unnamed_addr #0 {
/// bb3:
///   %0 = load atomic i64, ptr @_ZN38some_non_zero_from_atomic_optimization1X17h monotonic, align 8
///   %1 = icmp ne i64 %0, 0
///   tail call void @llvm.assume(i1 %1)
///   ret i64 %0
/// }
/// ```
///
/// The way we check that the LLVM IR is correct is by making sure that neither
/// `panic` nor `unreachable` is part of the LLVM IR:
// CHECK-LABEL: define {{.*}} i64 @some_non_zero_from_atomic_get2() {{.*}} {
// CHECK-NOT: panic
// CHECK-NOT: unreachable
#[no_mangle]
pub unsafe fn some_non_zero_from_atomic_get2() -> usize {
    match some_non_zero_from_atomic_get() {
        Some(x) => x.get(),
        None => unreachable!(), // shall be optimized out
    }
}
