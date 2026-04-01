// Evaluation of constants in array-elem count goes through different
// compiler control-flow paths.
//
// This test is checking the count in an array type.

//@ dont-require-annotations: NOTE

#![allow(unused_imports)]

const A_I8_T
    : [u32; (i8::MAX as i8 + 1u8) as usize]
    //~^ ERROR mismatched types
    //~| NOTE expected `i8`, found `u8`
    //~| ERROR cannot add `u8` to `i8`
    = [0; (i8::MAX as usize) + 1];


const A_CHAR_USIZE
    : [u32; 5u8 as char as usize]
    = [0; 5];


const A_BAD_CHAR_USIZE
    : [u32; 5i8 as char as usize]
    //~^ ERROR only `u8` can be cast as `char`, not `i8`
    = [0; 5];

fn main() {}
