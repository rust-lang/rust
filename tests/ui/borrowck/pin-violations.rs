//@ check-pass
// Test for pin violation detection - currently a placeholder
// The pin violation detection is implemented but may not trigger
// for all pin patterns yet. This test documents expected behavior.

#![feature(pin_ergonomics)]
#![allow(incomplete_features, dead_code)]

use std::marker::PhantomPinned;

#[derive(Default)]
struct Foo {
    _pinned: PhantomPinned,
}

// Test basic pin borrow patterns
fn test_pin_borrows() {
    let mut foo = Foo::default();
    let _pinned = &pin mut foo;
    // Currently, &pin mut creates a special borrow, not a Pin aggregate
    // So our violation detection may not trigger here
}

fn main() {}
