#![feature(coroutines, stmt_expr_attributes)]
#![allow(dropping_copy_types)]

//@ run-pass

use std::mem::size_of_val;

fn main() {
    // Coroutine taking a `Copy`able resume arg.
    let gen_copy = #[coroutine]
    |mut x: usize| {
        loop {
            drop(x);
            x = yield;
        }
    };

    // Coroutine taking a non-`Copy` resume arg.
    let gen_move = #[coroutine]
    |mut x: Box<usize>| {
        loop {
            drop(x);
            x = yield;
        }
    };

    // Neither of these coroutines have the resume arg live across the `yield`, so they should be
    // 1 Byte in size (only storing the discriminant)
    assert_eq!(size_of_val(&gen_copy), 1);
    assert_eq!(size_of_val(&gen_move), 1);
}
