//! Regression test for <https://github.com/rust-lang/rust/issues/60044>.

// We want to check that the None case is optimized away
//@ compile-flags: -O

// Simplify the emitted code
//@ compile-flags: -Cforce-unwind-tables=no

// Make it easier to write FileCheck directives:
//@ compile-flags: -Zmerge-functions=disabled

// So we don't have to worry about usize
//@ only-64bit

#![crate_type = "lib"]

use std::num::NonZeroUsize;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::Relaxed;

pub static X: AtomicUsize = AtomicUsize::new(1);

/// This function function shall look approximately like this:
/// ```llvm
/// define noundef range(i64 1, 0) i64 @some_non_zero_from_atomic_get() unnamed_addr #0 {
/// start:
///   %0 = load atomic i64, ptr @_ZN38some_non_zero_from_atomic_optimization1X13E monotonic, align 8
///   %1 = icmp ne i64 %0, 0
///   tail call void @llvm.assume(i1 %1)
///   ret i64 %0
/// }
/// ; ...
/// attributes #0 = { mustprogress nofree norecurse nounwind nonlazybind willreturn ... }
/// ```
// CHECK-LABEL: define {{.*}} i64 @some_non_zero_from_atomic_get()
// CHECK-SAME: #[[#ATTRIBUTE_GROUP:]] {
// CHECK: %0 = load atomic i64, ptr @{{[_a-zA-Z0-9]+}} monotonic, align 8
// CHECK-NEXT: %1 = icmp ne i64 %0, 0
// CHECK-NEXT: tail call void @llvm.assume(i1 %1)
// CHECK-NEXT: ret i64 %0
// CHECK-NEXT: }
#[no_mangle]
pub unsafe fn some_non_zero_from_atomic_get() -> Option<NonZeroUsize> {
    let x = X.load(Relaxed);
    Some(NonZeroUsize::new_unchecked(x))
}

/// This function shall be identical to the above:
// CHECK-LABEL: define {{.*}} i64 @some_non_zero_from_atomic_get2()
// CHECK-SAME: #[[#ATTRIBUTE_GROUP]] {
// CHECK: %0 = load atomic i64, ptr @{{[_a-zA-Z0-9]+}} monotonic, align 8
// CHECK-NEXT: %1 = icmp ne i64 %0, 0
// CHECK-NEXT: tail call void @llvm.assume(i1 %1)
// CHECK-NEXT: ret i64 %0
// CHECK-NEXT: }
#[no_mangle]
pub unsafe fn some_non_zero_from_atomic_get2() -> usize {
    match some_non_zero_from_atomic_get() {
        Some(x) => x.get(),
        None => unreachable!(), // shall be optimized out
    }
}

// Finally make sure the attribute group looks reasonable:
// CHECK-LABEL: attributes
// CHECK: #[[#ATTRIBUTE_GROUP]] =
// CHECK-SAME: mustprogress
// CHECK-SAME: nofree
// CHECK-SAME: norecurse
// CHECK-SAME: nounwind
// CHECK-SAME: willreturn
