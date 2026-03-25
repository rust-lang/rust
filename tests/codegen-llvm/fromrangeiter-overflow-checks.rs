// With -Coverflow-checks=yes (enabled by default by -Cdebug-assertions=yes) we will produce a
// runtime check that panics after yielding the maximum value of the range bound type. That is
// tested for by tests/ui/iterators/rangefrom-overflow-overflow-checks.rs
//
// This test ensures such runtime checks are optimized out when debug-assertions are
// enabled, but overflow-checks are explicitly disabled.

//@ revisions: DEBUG NOCHECKS
//@ compile-flags: -O -Cdebug-assertions=yes
//@ [NOCHECKS] compile-flags: -Coverflow-checks=no

#![crate_type = "lib"]
#![feature(new_range_api)]
use std::range::RangeFrom;

// CHECK-LABEL: @rangefrom_increments(
#[no_mangle]
pub unsafe fn rangefrom_increments(range: RangeFrom<i32>) -> RangeFrom<i32> {
    // Iterator is contained entirely within this function, so the optimizer should
    // be able to see that `exhausted` is never set and optimize out any branches.

    //         CHECK: i32 noundef {{(signext )?}}%range
    //         DEBUG: switch i32 %range
    //         DEBUG: call core::panicking::panic_const::panic_const_add_overflow
    //         DEBUG: unreachable
    //  NOCHECKS-NOT: unreachable
    //      NOCHECKS: [[REM:%[a-z_0-9.]+]] = add i32 %range, 2
    // NOCHECKS-NEXT: ret i32 [[REM]]

    let mut iter = range.into_iter();
    let _ = iter.next();
    let _ = iter.next();
    iter.remainder()
}
