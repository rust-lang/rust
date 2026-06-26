//@ build-pass

#![feature(covariant_unsafe_cell)]

use std::cell::CovariantUnsafeCell;

/// this function compiling ensures that CovariantUnsafeCell is actually covariant
fn _assert_covariance<'short, 'long: 'short>(
    x: CovariantUnsafeCell<&'long ()>,
) -> CovariantUnsafeCell<&'short ()> {
    x
}

fn main () {}
