// With -Coverflow-checks=yes (enabled by default by -Cdebug-assertions=yes) we will produce a
// runtime check that panics after yielding the maximum value of the range bound type. That is
// tested for by tests/ui/iterators/rangefrom-overflow-overflow-checks.rs
//
// This test ensures that such a runtime check is *not* emitted when debug-assertions are
// enabled, but overflow-checks are explicitly disabled.

//@ revisions: DEBUG NOCHECKS
//@ compile-flags: -O -Cdebug-assertions=yes
//@ [NOCHECKS] compile-flags: -Coverflow-checks=no

#![crate_type = "lib"]
#![feature(new_range_api)]
use std::range::{IterRangeFrom, RangeFrom};

// CHECK-LABEL: @iterrangefrom_remainder(
#[no_mangle]
pub unsafe fn iterrangefrom_remainder(x: IterRangeFrom<i32>) -> RangeFrom<i32> {
    //    DEBUG: i32 noundef %x.0
    // NOCHECKS: i32 noundef returned %x.0
    //    CHECK: i1 noundef zeroext %x.1
    //    DEBUG: br i1
    //    DEBUG: call core::panicking::panic_const::panic_const_add_overflow
    //    DEBUG: unreachable
    // NOCHECKS: ret i32 %x.0
    x.remainder()
}
