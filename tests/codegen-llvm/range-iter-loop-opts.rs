// This test ensures that Range iterators are optimizable, to
// the point that some loops can be entirely optimized out.

//@ compile-flags: -Copt-level=3

#![crate_type = "lib"]

use std::num::NonZeroU8;
use std::ops::{Range, RangeInclusive};

// CHECK-LABEL: @rangeinclusive_noop_loop = unnamed_addr alias void (), ptr @range_noop_loop
// CHECK-LABEL: @rangeinclusive_nz_noop_loop = unnamed_addr alias void (), ptr @range_noop_loop

// CHECK-LABEL: @range_noop_loop(
#[no_mangle]
pub unsafe fn range_noop_loop() {
    // CHECK-NEXT: start:
    // CHECK-NEXT: ret void

    // This loop should be optimized out entirely.
    for _ in 0_u8..100 {
        ()
    }
}

// CHECK-LABEL: @range_count(
#[no_mangle]
pub unsafe fn range_count(s: u8, e: u8) -> usize {
    //  CHECK-NOT: br {{.*}}
    //      CHECK: ret i{{8|16|32|64}}

    // This loop should be optimized to arithmetic.
    let mut count = 0;
    for _ in s..e {
        count += 1;
    }
    count
}

// Deduplicated to alias of range_noop_loop, checked above
#[no_mangle]
pub unsafe fn rangeinclusive_noop_loop() {
    // This loop should be optimized out entirely.
    for _ in 0_u8..=100 {
        ()
    }
}

// CHECK-LABEL: @rangeinclusive_count(
#[no_mangle]
pub unsafe fn rangeinclusive_count(s: u8, e: u8) -> usize {
    //  CHECK-NOT: br {{.*}}
    //      CHECK: ret i{{8|16|32|64}}

    // This loop should be optimized to arithmetic.
    let mut count = 0;
    for _ in s..=e {
        count += 1;
    }
    count
}

// Deduplicated to alias of range_noop_loop, checked above
#[no_mangle]
pub unsafe fn rangeinclusive_nz_noop_loop() {
    // This loop should be optimized out entirely.
    for _ in NonZeroU8::new(1).unwrap()..=NonZeroU8::new(100).unwrap() {
        ()
    }
}

// CHECK-LABEL: @rangeinclusive_nz_count(
#[no_mangle]
pub unsafe fn rangeinclusive_nz_count(s: NonZeroU8, e: NonZeroU8) -> usize {
    //      CHECK: br {{.*}}
    //      CHECK: ret i{{8|16|32|64}}

    // RangeInclusive<NonZero> cannot optimize the same way
    // because Step::forward_overflowing on NonZero cannot
    // be allowed to wrap to 0.
    let mut count = 0;
    for _ in s..=e {
        count += 1;
    }
    count
}
