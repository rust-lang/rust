#![warn(clippy::missing_const_for_fn)]
#![allow(incomplete_features, clippy::let_and_return)]
#![feature(const_generics)]

use std::mem::transmute;

struct Game {
    guess: i32,
}

impl Game {
    // Could be const
    pub fn new() -> Self {
        Self { guess: 42 }
    }

    fn const_generic_params<'a, T, const N: usize>(&self, b: &'a [T; N]) -> &'a [T; N] {
        b
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

// Could be const (since Rust 1.39)
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

fn sub(x: u32) -> usize {
    unsafe { transmute(&x) }
}

// NOTE: This is currently not yet allowed to be const
// Once implemented, Clippy should be able to suggest this as const, too.
fn generic_arr<T: Copy>(t: [T; 1]) -> T {
    t[0]
}

mod with_drop {
    pub struct A;
    pub struct B;
    impl Drop for A {
        fn drop(&mut self) {}
    }

    impl B {
        // This can be const, because `a` is passed by reference
        pub fn b(self, a: &A) -> B {
            B
        }
    }
}

// Should not be const
fn main() {}
