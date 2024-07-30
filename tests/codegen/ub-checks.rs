// With -Zub-checks=yes (enabled by default by -Cdebug-assertions=yes) we will produce a runtime
// check that the index to slice::get_unchecked is in-bounds of the slice. That is tested for by
// tests/ui/precondition-checks/out-of-bounds-get-unchecked.rs
//
// This test ensures that such a runtime check is *not* emitted when debug-assertions are enabled,
// but ub-checks are explicitly disabled.

//@ revisions: debug nochecks
//@ [debug] compile-flags:
//@ [nochecks] compile-flags: -Zub-checks=no
//@ compile-flags: -O -Cdebug-assertions=yes

#![crate_type = "lib"]

use std::ops::Range;

// CHECK-LABEL: @slice_get_unchecked(
#[no_mangle]
pub unsafe fn slice_get_unchecked(x: &[i32], i: usize) -> &i32 {
    //          CHECK: icmp ult
    // CHECK-NOCHECKS: tail call void @llvm.assume
    //    CHECK-DEBUG: br i1
    //    CHECK-DEBUG: call core::panicking::panic_nounwind
    //    CHECK-DEBUG: unreachable
    //          CHECK: getelementptr inbounds
    //          CHECK: ret ptr
    x.get_unchecked(i)
}
