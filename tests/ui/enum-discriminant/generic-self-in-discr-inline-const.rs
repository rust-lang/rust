//@ check-fail
// Test that `Self` is rejected even when nested inside an inline const
// or closure within an enum discriminant. Regression test for issue #154281.
#![feature(sized_hierarchy)]

use std::marker::PointeeSized;

#[repr(usize)]
enum What<T: PointeeSized> {
    X = const { { let _: *mut Self; 1_usize } },
    //~^ ERROR generic `Self` types are not permitted in enum discriminant values
    Y(*mut T),
}

fn main() {}
