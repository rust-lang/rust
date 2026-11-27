// With -Zub-checks=yes (enabled by default by -Cdebug-assertions=yes) we will produce a runtime
// check that the index to slice::get_unchecked is in-bounds of the slice. That is tested for by
// tests/ui/precondition-checks/out-of-bounds-get-unchecked.rs
//
// This test ensures that such a runtime check is *not* emitted when debug-assertions are enabled,
// but ub-checks are explicitly disabled.

//@ revisions: DEBUG NOCHECKS
// [DEBUG] no extra compile-flags
//@ [NOCHECKS] compile-flags: -Zub-checks=no
//@ compile-flags: -Copt-level=3 -Cdebug-assertions=yes

#![crate_type = "lib"]

use std::ops::Range;

// CHECK-LABEL: @slice_get_unchecked(
#[no_mangle]
pub unsafe fn slice_get_unchecked(x: &[i32], i: usize) -> &i32 {
    //    CHECK: icmp ult
    // NOCHECKS: tail call void @llvm.assume
    //    DEBUG: br i1
    //    DEBUG: call core::panicking::panic_nounwind
    //    DEBUG: unreachable
    //    CHECK: getelementptr inbounds
    //    CHECK: ret ptr
    x.get_unchecked(i)
}
