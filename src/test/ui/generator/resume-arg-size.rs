#![feature(generators)]

// run-pass

use std::mem::size_of_val;

fn main() {
    // Generator taking a `Copy`able resume arg.
    let gen_copy = |mut x: usize| {
        loop {
            drop(x);
            x = yield;
        }
    };

    // Generator taking a non-`Copy` resume arg.
    let gen_move = |mut x: Box<usize>| {
        loop {
            drop(x);
            x = yield;
        }
    };

    // Neither of these generators have the resume arg live across the `yield`, so they should be
    // 1 Byte in size (only storing the discriminant)
    assert_eq!(size_of_val(&gen_copy), 1);
    assert_eq!(size_of_val(&gen_move), 1);
}
