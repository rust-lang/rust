//@ compile-flags: -Cno-prepopulate-passes -Copt-level=0 -Cdebug-assertions=no

// This test ensures that in a debug build which turns off debug assertions, we do not monomorphize
// any of the standard library's unsafe precondition checks.
// The naive codegen of those checks contains the actual check underneath an `if false`, which
// could be optimized out if optimizations are enabled. But if we rely on optimizations to remove
// panic branches, then we can't link compiler_builtins without optimizing it, which means that
// -Zbuild-std doesn't work with -Copt-level=0.
//
// In other words, this tests for a mandatory optimization.

#![crate_type = "lib"]

use std::ptr::NonNull;

// CHECK-LABEL: ; core::ptr::non_null::NonNull<T>::new_unchecked
// CHECK-NOT: call
// CHECK: }

// CHECK-LABEL: @nonnull_new
#[no_mangle]
pub unsafe fn nonnull_new(ptr: *mut u8) -> NonNull<u8> {
    // CHECK: ; call core::ptr::non_null::NonNull<T>::new_unchecked
    unsafe { NonNull::new_unchecked(ptr) }
}
