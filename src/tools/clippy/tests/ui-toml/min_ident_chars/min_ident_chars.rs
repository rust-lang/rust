//@aux-build:extern_types.rs
#![allow(nonstandard_style, unused)]
#![warn(clippy::min_ident_chars)]

extern crate extern_types;
use extern_types::{Aaa, LONGER, M, N as W};
//~^ min_ident_chars

pub const N: u32 = 0;
//~^ min_ident_chars
pub const LONG: u32 = 32;

struct Owo {
    Uwu: u128,
    aaa: Aaa,
    //~^ min_ident_chars
}

fn main() {
    let wha = 1;
    let vvv = 1;
    //~^ min_ident_chars
    let uuu = 1;
    //~^ min_ident_chars
    let (mut a, mut b) = (1, 2);
    //~^ min_ident_chars
    //~| min_ident_chars
    for i in 0..1000 {}
    //~^ min_ident_chars
}
