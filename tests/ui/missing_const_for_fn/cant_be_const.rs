//! False-positive tests to ensure we don't suggest `const` for things where it would cause a
//! compilation error.
//! The .stderr output of this test should be empty. Otherwise it's a bug somewhere.

#![warn(clippy::missing_const_for_fn)]
#![feature(start)]

struct Game;

// This should not be linted because it's already const
const fn already_const() -> i32 { 32 }

impl Game {
    // This should not be linted because it's already const
    pub const fn already_const() -> i32 { 32 }
}

// Allowing on this function, because it would lint, which we don't want in this case.
#[allow(clippy::missing_const_for_fn)]
fn random() -> u32 { 42 }

// We should not suggest to make this function `const` because `random()` is non-const
fn random_caller() -> u32 {
    random()
}

static Y: u32 = 0;

// We should not suggest to make this function `const` because const functions are not allowed to
// refer to a static variable
fn get_y() -> u32 {
    Y
        //~^ ERROR E0013
}

// Also main should not be suggested to be made const
fn main() {
    // We should also be sure to not lint on closures
    let add_one_v2 = |x: u32| -> u32 { x + 1 };
}

trait Foo {
    // This should not be suggested to be made const
    // (rustc restriction)
    fn f() -> u32;
}

// Don't lint custom entrypoints either
#[start]
fn init(num: isize, something: *const *const u8) -> isize { 1 }
