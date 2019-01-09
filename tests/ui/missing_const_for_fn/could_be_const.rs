#![warn(clippy::missing_const_for_fn)]
#![allow(clippy::let_and_return)]

use std::mem::transmute;

struct Game {
    guess: i32,
}

impl Game {
    // Could be const
    pub fn new() -> Self {
        Self {
            guess: 42,
        }
    }
}

// Could be const
fn one() -> i32 { 1 }

// Could also be const
fn two() -> i32 {
    let abc = 2;
    abc
}

// TODO: Why can this be const? because it's a zero sized type?
// There is the `const_string_new` feature, but it seems that this already works in const fns?
fn string() -> String {
    String::new()
}

// Could be const
unsafe fn four() -> i32 { 4 }

// Could also be const
fn generic<T>(t: T) -> T {
    t
}

// FIXME: This could be const but is currently not linted
fn sub(x: u32) -> usize {
    unsafe { transmute(&x) }
}

// FIXME: This could be const but is currently not linted
fn generic_arr<T: Copy>(t: [T; 1]) -> T {
    t[0]
}

// Should not be const
fn main() {}
