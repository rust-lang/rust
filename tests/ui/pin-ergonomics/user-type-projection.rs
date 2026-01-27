#![crate_type = "rlib"]
#![feature(pin_ergonomics)]
#![expect(incomplete_features)]
//@ edition: 2024
//@ check-pass

// Test that we don't ICE when projecting user-type-annotations through a `&pin` pattern.
//
// Historically, this could occur when the code handling those projections did not know
// about `&pin` patterns, and incorrectly treated them as plain `&`/`&mut` patterns instead.

struct Data {
    x: u32
}

pub fn project_user_type_through_pin() -> u32 {
    let &pin const Data { x }: &pin const Data = &pin const Data { x: 30 };
    x
}
