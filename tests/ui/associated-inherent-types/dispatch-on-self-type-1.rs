//@ check-pass

#![feature(inherent_associated_types, auto_traits, negative_impls)]
#![allow(incomplete_features)]

use std::cmp::Ordering;

// Check that inherent associated types are dispatched on the concrete Self type.

struct Select<T, U>(T, U);

impl<T: Ordinary, U: Ordinary> Select<T, U> {
    type Type = ();
}

impl<T: Ordinary> Select<T, Special> {
    type Type = bool;
}

impl<T: Ordinary> Select<Special, T> {
    type Type = Ordering;
}

impl Select<Special, Special> {
    type Type = (bool, bool);
}

fn main() {
    let _: Select<String, Special>::Type = false;
    let _: Select<Special, Special>::Type = (true, false);
    let _: Select<Special, u8>::Type = Ordering::Equal;
    let _: Select<i128, ()>::Type = ();
}

enum Special {}

impl !Ordinary for Special {}

auto trait Ordinary {}
