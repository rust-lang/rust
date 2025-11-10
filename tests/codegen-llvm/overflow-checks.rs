// With -Coverflow-checks=yes (enabled by default by -Cdebug-assertions=yes) we will produce a
// runtime check that panics when an operation would result in integer overflow.
//
// This test ensures that such a runtime check is *not* emitted when debug-assertions are enabled,
// but overflow-checks are explicitly disabled. It also ensures that even if a dependency is
// compiled with overflow checks, `intrinsics::overflow_checks()` will be treated with the
// overflow-checks setting of the current crate (when `#[rustc_inherit_overflow_checks]`) is used.

//@ aux-build:overflow_checks_add.rs
//@ revisions: DEBUG NOCHECKS
//@ compile-flags: -O -Cdebug-assertions=yes
//@ [NOCHECKS] compile-flags: -Coverflow-checks=no

#![crate_type = "lib"]

extern crate overflow_checks_add;

// CHECK-LABEL: @add(
#[no_mangle]
pub unsafe fn add(a: u8, b: u8) -> u8 {
    //        CHECK: i8 noundef %a, i8 noundef %b
    //        CHECK: add i8 %b, %a
    //        DEBUG: icmp ult i8 [[zero:[^,]+]], %a
    //        DEBUG: call core::num::overflow_panic::add
    //        DEBUG: unreachable
    // NOCHECKS-NOT: unreachable
    //     NOCHECKS: ret i8 %0
    overflow_checks_add::add(a, b)
}
