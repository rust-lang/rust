// run-rustfix

#![warn(clippy::manual_bits)]
#![allow(clippy::no_effect, path_statements, unused_must_use, clippy::unnecessary_operation)]

use std::mem::{size_of, size_of_val};

fn main() {
    size_of::<i8>() * 8;
    size_of::<i16>() * 8;
    size_of::<i32>() * 8;
    size_of::<i64>() * 8;
    size_of::<i128>() * 8;
    size_of::<isize>() * 8;

    size_of::<u8>() * 8;
    size_of::<u16>() * 8;
    size_of::<u32>() * 8;
    size_of::<u64>() * 8;
    size_of::<u128>() * 8;
    size_of::<usize>() * 8;

    8 * size_of::<i8>();
    8 * size_of::<i16>();
    8 * size_of::<i32>();
    8 * size_of::<i64>();
    8 * size_of::<i128>();
    8 * size_of::<isize>();

    8 * size_of::<u8>();
    8 * size_of::<u16>();
    8 * size_of::<u32>();
    8 * size_of::<u64>();
    8 * size_of::<u128>();
    8 * size_of::<usize>();

    size_of::<usize>() * 4;
    4 * size_of::<usize>();
    size_of::<bool>() * 8;
    8 * size_of::<bool>();

    size_of_val(&0u32) * 8;

    type Word = u32;
    size_of::<Word>() * 8;
    type Bool = bool;
    size_of::<Bool>() * 8;
}
