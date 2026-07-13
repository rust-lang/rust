//@ compile-flags: -O
//@ ignore-std-debug-assertions
//@ only-64bit
#![crate_type = "lib"]
#![feature(exact_size_is_empty)]

use std::range::{RangeInclusive, RangeInclusiveIter};

// Check that a for loop over the new `..=` optimizes to the obvious loop
#[no_mangle]
pub fn every_fencepost(slice: &[u8]) {
    // CHECK-LABEL: @every_fencepost

    // CHECK: start:
    // CHECK-NEXT: br label %[[LOOP:.+$]]

    // CHECK: [[LOOP]]:
    // CHECK-NEXT: [[I:%.+]] = phi i64 [ 0, %start ], [ [[NEXT_I:%.+]], %[[LOOP]] ]
    // CHECK-NEXT: [[NEXT_I]] = add nuw i64 [[I]], 1
    // CHECK-NEXT: call void @do_something(i64{{.*}} [[I]])
    // CHECK-NEXT: [[DONE:%.+]] = icmp eq i64 [[I]], %slice.1
    // CHECK-NEXT: br i1 [[DONE]], label %[[EXIT:.+]], label %[[LOOP]]

    // CHECK: [[EXIT]]:
    // CHECK-NEXT: ret void

    for i in (RangeInclusive { start: 0, last: slice.len() }) {
        do_something(i)
    }
}

unsafe extern "Rust" {
    safe fn do_something(_: usize);
}

// Ensure that, despite the pre-processing done in `into_iter`, simple things
// still optimize down to simple operations.
#[no_mangle]
pub fn make_ord_iter_check_empty(first: u8, last: u8) -> bool {
    // CHECK-LABEL: @make_ord_iter_check_empty
    // CHECK: [[RET:%.+]] = icmp ugt i8 %first, %last
    // CHECK: ret i1 [[RET]]
    RangeInclusive { start: first, last }.into_iter().is_empty()
}

// Ensure that for an `Ord` type (here `u64`) there's only one check needed for this.
// AKA that the second check (needed for `PartialOrd`-only things) is optimized out.
#[no_mangle]
pub fn make_ord_iter(first: u64, last: u64) -> RangeInclusiveIter<u64> {
    // CHECK-LABEL: @make_ord_iter
    // CHECK: start:
    // CHECK-NEXT: [[NEEDS_EXCLUSIVE:%.+]] = icmp eq i64 %last, -1
    // CHECK-NEXT: [[LAST_P1:%.+]] = add nuw i64 %last, 1
    // CHECK-NEXT: [[END:%.+]] = select i1 [[NEEDS_EXCLUSIVE]], i64 -1, i64 [[LAST_P1]]
    // CHECK-NEXT: [[IS_EXCLUSIVE:%.+]] = zext i1 [[NEEDS_EXCLUSIVE]] to i8
    // CHECK-NOT: store
    // CHECK: store i64 %first,
    // CHECK-NOT: store
    // CHECK: store i64 [[END]],
    // CHECK-NOT: store
    // CHECK: store i8 [[IS_EXCLUSIVE]],
    // CHECK-NOT: store
    // CHECK: ret void
    RangeInclusive { start: first, last }.into_iter()
}
