// run-rustfix
// aux-build:proc_macros.rs

#![warn(clippy::manual_rem_euclid)]
#![allow(clippy::let_with_type_underscore)]

extern crate proc_macros;
use proc_macros::{external, inline_macros};

#[inline_macros]
fn main() {
    let value: i32 = 5;

    let _: i32 = ((value % 4) + 4) % 4;
    let _: i32 = (4 + (value % 4)) % 4;
    let _: i32 = (value % 4 + 4) % 4;
    let _: i32 = (4 + value % 4) % 4;
    let _: i32 = 1 + (4 + value % 4) % 4;

    let _: i32 = (3 + value % 4) % 4;
    let _: i32 = (-4 + value % -4) % -4;
    let _: i32 = ((5 % 4) + 4) % 4;

    // Make sure the lint does not trigger if it would cause an error, like with an ambiguous
    // integer type
    let not_annotated = 24;
    let _ = ((not_annotated % 4) + 4) % 4;
    let inferred: _ = 24;
    let _ = ((inferred % 4) + 4) % 4;

    // For lint to apply the constant must always be on the RHS of the previous value for %
    let _: i32 = 4 % ((value % 4) + 4);
    let _: i32 = ((4 % value) + 4) % 4;

    // Lint in internal macros
    inline!(
        let value: i32 = 5;
        let _: i32 = ((value % 4) + 4) % 4;
    );

    // Do not lint in external macros
    external!(
        let value: i32 = 5;
        let _: i32 = ((value % 4) + 4) % 4;
    );
}

// Should lint for params too
pub fn rem_euclid_4(num: i32) -> i32 {
    ((num % 4) + 4) % 4
}

// Constant version came later, should still lint
pub const fn const_rem_euclid_4(num: i32) -> i32 {
    ((num % 4) + 4) % 4
}

#[clippy::msrv = "1.37"]
pub fn msrv_1_37() {
    let x: i32 = 10;
    let _: i32 = ((x % 4) + 4) % 4;
}

#[clippy::msrv = "1.38"]
pub fn msrv_1_38() {
    let x: i32 = 10;
    let _: i32 = ((x % 4) + 4) % 4;
}

// For const fns:
#[clippy::msrv = "1.51"]
pub const fn msrv_1_51() {
    let x: i32 = 10;
    let _: i32 = ((x % 4) + 4) % 4;
}

#[clippy::msrv = "1.52"]
pub const fn msrv_1_52() {
    let x: i32 = 10;
    let _: i32 = ((x % 4) + 4) % 4;
}
