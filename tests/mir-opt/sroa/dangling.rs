//@ test-mir-pass: ScalarReplacementOfAggregates
// Regression test for https://github.com/rust-lang/rust/issues/164176.
#![crate_type = "lib"]
#![feature(maybe_dangling)]
use std::mem::{ManuallyDrop, MaybeDangling};

#[derive(Copy, Clone)]
pub struct S<T>(T);

// Check that SROA does optimize the general pattern used in subsequent negative tests.
//
// CHECK-LABEL: fn positive(
// CHECK: debug a => [[A:_.+]];
// CHECK: debug ((b: S<T>).0: T) => [[B:_.+]];
// CHECK: [[B]] = copy ([[A]].0: T);
pub fn positive<T: Copy>(a: S<T>) {
    let b = a;
}

// References inside a MaybeDangling doesn't have to satisfy any memory-depended validity
// properties. SROA cannot rewrite operations in terms of an inner type, which might require it.
//
// CHECK-LABEL: fn maybe_dangling(
// CHECK: debug a => [[A:_.+]];
// CHECK: debug b => [[B:_.+]];
// CHECK: [[B]] = copy [[A]];
pub fn maybe_dangling<T: Copy>(a: MaybeDangling<T>) {
    let b = a;
}

// ManuallyDrop is like MaybeDangling.
//
// CHECK-LABEL: fn manually_drop(
// CHECK: debug a => [[A:_.+]];
// CHECK: debug b => [[B:_.+]];
// CHECK: [[B]] = move [[A]];
pub fn manually_drop<T>(a: ManuallyDrop<T>) {
    let b = a;
}
