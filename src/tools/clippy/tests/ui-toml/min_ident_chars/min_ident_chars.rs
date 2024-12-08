//@aux-build:extern_types.rs
#![allow(nonstandard_style, unused)]
#![warn(clippy::min_ident_chars)]

extern crate extern_types;
use extern_types::{Aaa, LONGER, M, N as W};

pub const N: u32 = 0;
pub const LONG: u32 = 32;

struct Owo {
    Uwu: u128,
    aaa: Aaa,
}

fn main() {
    let wha = 1;
    let vvv = 1;
    let uuu = 1;
    let (mut a, mut b) = (1, 2);
    for i in 0..1000 {}
}
