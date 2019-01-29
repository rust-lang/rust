#![warn(clippy::missing_const_for_fn)]
#![allow(clippy::let_and_return)]

use std::mem::transmute;

struct Game {
    guess: i32,
}

impl Game {
    // Could be const
    pub fn new() -> Self {
        Self { guess: 42 }
    }
}

// Could be const
fn one() -> i32 {
    1
}

// Could also be const
fn two() -> i32 {
    let abc = 2;
    abc
}

// FIXME: This is a false positive in the `is_min_const_fn` function.
// At least until the `const_string_new` feature is stabilzed.
fn string() -> String {
    String::new()
}

// Could be const
unsafe fn four() -> i32 {
    4
}

// Could also be const
fn generic<T>(t: T) -> T {
    t
}

// FIXME: Depends on the `const_transmute` and `const_fn` feature gates.
// In the future Clippy should be able to suggest this as const, too.
fn sub(x: u32) -> usize {
    unsafe { transmute(&x) }
}

// NOTE: This is currently not yet allowed to be const
// Once implemented, Clippy should be able to suggest this as const, too.
fn generic_arr<T: Copy>(t: [T; 1]) -> T {
    t[0]
}

// Should not be const
fn main() {}
