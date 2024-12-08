//@ run-pass
// Test for issue #112204 -- make sure this goes through the entire compilation pipeline,
// similar to why `offset-of-unsized.rs` is also build-pass

use std::mem::offset_of;

type ComplexTup = ((u8, (u8, (u8, u16), u8)), (u8, u32, u16));

fn main() {
    println!("{}", offset_of!(((u8, u8), u8), 0));
    println!("{}", offset_of!(((u8, u8), u8), 1));
    println!("{}", offset_of!(((u8, (u8, u8)), (u8, u8, u8)), 0.1.0));

    // Complex case: do all combinations of spacings because the spacing determines what gets
    // sent to the lexer.
    println!("{}", offset_of!(ComplexTup, 0.1.1.1));
    println!("{}", offset_of!(ComplexTup, 0. 1.1.1));
    println!("{}", offset_of!(ComplexTup, 0 . 1.1.1));
    println!("{}", offset_of!(ComplexTup, 0 .1.1.1));
    println!("{}", offset_of!(ComplexTup, 0.1 .1.1));
    println!("{}", offset_of!(ComplexTup, 0.1 . 1.1));
    println!("{}", offset_of!(ComplexTup, 0.1. 1.1));
    println!("{}", offset_of!(ComplexTup, 0.1.1. 1));
    println!("{}", offset_of!(ComplexTup, 0.1.1 . 1));
    println!("{}", offset_of!(ComplexTup, 0.1.1 .1));

    println!("{}", offset_of!(((u8, u16), (u32, u16, u8)), 0.0));
    println!("{}", offset_of!(((u8, u16), (u32, u16, u8)), 1.2));
}
