#![feature(plugin)]
#![plugin(clippy)]

#![allow(unused)]

#[deny(absurd_unsigned_comparisons)]
fn main() {
    1u32 <= 0; //~ERROR testing whether an unsigned integer is non-positive
    1u8 <= 0; //~ERROR testing whether an unsigned integer is non-positive
    1i32 <= 0;
    0 >= 1u32; //~ERROR testing whether an unsigned integer is non-positive
    0 >= 1;
    1u32 > 0;
}
