//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)

//! Regression test for https://github.com/rust-lang/rust/issues/62211
//!
//! The old implementation of defaults did not check whether the provided
//! default actually fulfills all bounds on the assoc. type, leading to
//! unsoundness, demonstrated here as a use-after-free.
//!
//! Note that the underlying cause of this is still not yet fixed.
//! See: https://github.com/rust-lang/rust/issues/33017

#![feature(associated_type_defaults)]

use std::{
    fmt::Display,
    ops::{AddAssign, Deref},
};

trait UncheckedCopy: Sized {
    // This Output is said to be Copy. Yet we default to Self
    // and it's accepted, not knowing if Self ineed is Copy
    type Output: Copy + Deref<Target = str> + AddAssign<&'static str> + From<Self> + Display = Self;
    //~^ ERROR the trait bound `Self: Copy` is not satisfied
    //~| ERROR the trait bound `Self: Deref` is not satisfied
    //~| ERROR cannot add-assign `&'static str` to `Self`
    //~| ERROR `Self` doesn't implement `std::fmt::Display`

    // We said the Output type was Copy, so we can Copy it freely!
    fn unchecked_copy(other: &Self::Output) -> Self::Output {
        (*other)
    }

    fn make_origin(s: Self) -> Self::Output {
        s.into()
    }
}

impl<T> UncheckedCopy for T {}

fn bug<T: UncheckedCopy>(origin: T) {
    let origin = T::make_origin(origin);
    let mut copy = T::unchecked_copy(&origin);

    // assert we indeed have 2 strings pointing to the same buffer.
    assert_eq!(origin.as_ptr(), copy.as_ptr());

    // Drop the origin. Any use of `copy` is UB.
    drop(origin);

    copy += "This is invalid!";
    println!("{}", copy);
}

fn main() {
    bug(String::from("hello!"));
}
