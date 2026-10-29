//@aux-build:proc_macros.rs

#![warn(clippy::bad_bit_mask)]
#![expect(
    unused_comparisons,
    clippy::absurd_extreme_comparisons,
    clippy::bad_bit_mask,
    clippy::identity_op
)]

use core::hint::black_box;
use proc_macros::{external, with_span};

fn main() {
    let x = black_box(5u32);

    let _ = x | 0b00_0000 > 0b00_0000; //~ ineffective_bit_mask
    let _ = x | 0b00_0001 > 0b00_0000;
    let _ = x | 0b00_0001 > 0b00_0001; //~ ineffective_bit_mask
    let _ = x | 0b00_0010 > 0b00_0001;
    let _ = x | 0b00_0011 > 0b00_0001;
    let _ = x | 0b00_0000 > 0b00_0011; //~ ineffective_bit_mask
    let _ = x | 0b00_0001 > 0b00_0011; //~ ineffective_bit_mask
    let _ = x | 0b00_0010 > 0b00_0011; //~ ineffective_bit_mask
    let _ = x | 0b00_0011 > 0b00_0011; //~ ineffective_bit_mask
    let _ = x | 0b00_0100 > 0b00_0011;
    let _ = x | 0b00_0101 > 0b00_0011;
    let _ = x | 0b00_0001 > 0b00_0111; //~ ineffective_bit_mask
    let _ = x | 0b00_0010 > 0b00_0111; //~ ineffective_bit_mask
    let _ = x | 0b00_0011 > 0b00_0111; //~ ineffective_bit_mask
    let _ = x | 0b00_0100 > 0b00_0111; //~ ineffective_bit_mask
    let _ = x | 0b00_1000 > 0b00_0111;
    let _ = x | 0b00_1010 > 0b00_0111;
    let _ = x | 0b00_0001 > 0b00_1111; //~ ineffective_bit_mask
    let _ = x | 0b00_0010 > 0b00_1111; //~ ineffective_bit_mask
    let _ = x | 0b00_0011 > 0b00_1111; //~ ineffective_bit_mask
    let _ = x | 0b00_0100 > 0b00_1111; //~ ineffective_bit_mask
    let _ = x | 0b00_0101 > 0b00_1111; //~ ineffective_bit_mask
    let _ = x | 0b00_0110 > 0b00_1111; //~ ineffective_bit_mask
    let _ = x | 0b00_0111 > 0b00_1111; //~ ineffective_bit_mask
    let _ = x | 0b00_1000 > 0b00_1111; //~ ineffective_bit_mask
    let _ = x | 0b00_1001 > 0b00_1111; //~ ineffective_bit_mask
    let _ = x | 0b00_1111 > 0b10_1111; //~ ineffective_bit_mask
    let _ = x | 0b01_0000 > 0b10_1111;
    let _ = x | 0b01_1111 > 0b10_1111;
    let _ = x | 0b10_0000 > 0b10_1111;
    let _ = x | 0b11_1111 > 0b10_1111;
    let _ = x | 0b00_1111 <= 0b10_1111; //~ ineffective_bit_mask
    let _ = x | 0b11_0000 <= 0b10_1111;

    let _ = x ^ 0b00_0000 > 0b00_0000; //~ ineffective_bit_mask
    let _ = x ^ 0b00_0001 > 0b00_0000;
    let _ = x ^ 0b00_0001 > 0b00_0001; //~ ineffective_bit_mask
    let _ = x ^ 0b00_0010 > 0b00_0001;
    let _ = x ^ 0b00_0011 > 0b00_0001;
    let _ = x ^ 0b00_0000 > 0b00_0011; //~ ineffective_bit_mask
    let _ = x ^ 0b00_0001 > 0b00_0011; //~ ineffective_bit_mask
    let _ = x ^ 0b00_0010 > 0b00_0011; //~ ineffective_bit_mask
    let _ = x ^ 0b00_0011 > 0b00_0011; //~ ineffective_bit_mask
    let _ = x ^ 0b00_0100 > 0b00_0011;
    let _ = x ^ 0b00_0101 > 0b00_0011;
    let _ = x ^ 0b00_0001 > 0b00_0111; //~ ineffective_bit_mask
    let _ = x ^ 0b00_0010 > 0b00_0111; //~ ineffective_bit_mask
    let _ = x ^ 0b00_0011 > 0b00_0111; //~ ineffective_bit_mask
    let _ = x ^ 0b00_0100 > 0b00_0111; //~ ineffective_bit_mask
    let _ = x ^ 0b00_1000 > 0b00_0111;
    let _ = x ^ 0b00_1010 > 0b00_0111;
    let _ = x ^ 0b00_0001 > 0b00_1111; //~ ineffective_bit_mask
    let _ = x ^ 0b00_0010 > 0b00_1111; //~ ineffective_bit_mask
    let _ = x ^ 0b00_0011 > 0b00_1111; //~ ineffective_bit_mask
    let _ = x ^ 0b00_0100 > 0b00_1111; //~ ineffective_bit_mask
    let _ = x ^ 0b00_0101 > 0b00_1111; //~ ineffective_bit_mask
    let _ = x ^ 0b00_0110 > 0b00_1111; //~ ineffective_bit_mask
    let _ = x ^ 0b00_0111 > 0b00_1111; //~ ineffective_bit_mask
    let _ = x ^ 0b00_1000 > 0b00_1111; //~ ineffective_bit_mask
    let _ = x ^ 0b00_1001 > 0b00_1111; //~ ineffective_bit_mask
    let _ = x ^ 0b00_1111 > 0b10_1111; //~ ineffective_bit_mask
    let _ = x ^ 0b01_0000 > 0b10_1111;
    let _ = x ^ 0b01_1111 > 0b10_1111;
    let _ = x ^ 0b10_0000 > 0b10_1111;
    let _ = x ^ 0b11_1111 > 0b10_1111;
    let _ = x ^ 0b00_1111 <= 0b10_1111; //~ ineffective_bit_mask
    let _ = x ^ 0b11_0000 <= 0b10_1111;

    let _ = x | 0b00_0000 < 0b00_0000;
    let _ = x | 0b00_0001 < 0b00_0000;
    let _ = x | 0b00_0001 < 0b00_0001;
    let _ = x | 0b00_0010 < 0b00_0001;
    let _ = x | 0b00_0011 < 0b00_0001;
    let _ = x | 0b00_0000 < 0b00_0010; //~ ineffective_bit_mask
    let _ = x | 0b00_0001 < 0b00_0010; //~ ineffective_bit_mask
    let _ = x | 0b00_0010 < 0b00_0010;
    let _ = x | 0b00_0011 < 0b00_0010;
    let _ = x | 0b00_0000 < 0b00_0100; //~ ineffective_bit_mask
    let _ = x | 0b00_0001 < 0b00_0100; //~ ineffective_bit_mask
    let _ = x | 0b00_0010 < 0b00_0100; //~ ineffective_bit_mask
    let _ = x | 0b00_0011 < 0b00_0100; //~ ineffective_bit_mask
    let _ = x | 0b00_0100 < 0b00_0100;
    let _ = x | 0b00_0101 < 0b00_0100;
    let _ = x | 0b00_0001 < 0b00_1000; //~ ineffective_bit_mask
    let _ = x | 0b00_0010 < 0b00_1000; //~ ineffective_bit_mask
    let _ = x | 0b00_0011 < 0b00_1000; //~ ineffective_bit_mask
    let _ = x | 0b00_0100 < 0b00_1000; //~ ineffective_bit_mask
    let _ = x | 0b00_1000 < 0b00_1000;
    let _ = x | 0b00_1010 < 0b00_1000;
    let _ = x | 0b00_0001 < 0b01_0000; //~ ineffective_bit_mask
    let _ = x | 0b00_0010 < 0b01_0000; //~ ineffective_bit_mask
    let _ = x | 0b00_0011 < 0b01_0000; //~ ineffective_bit_mask
    let _ = x | 0b00_0100 < 0b01_0000; //~ ineffective_bit_mask
    let _ = x | 0b00_0101 < 0b01_0000; //~ ineffective_bit_mask
    let _ = x | 0b00_0110 < 0b01_0000; //~ ineffective_bit_mask
    let _ = x | 0b00_0111 < 0b01_0000; //~ ineffective_bit_mask
    let _ = x | 0b00_1000 < 0b01_0000; //~ ineffective_bit_mask
    let _ = x | 0b00_1001 < 0b01_0000; //~ ineffective_bit_mask
    let _ = x | 0b00_1111 < 0b11_0000; //~ ineffective_bit_mask
    let _ = x | 0b01_0000 < 0b11_0000;
    let _ = x | 0b01_1111 < 0b11_0000;
    let _ = x | 0b10_0000 < 0b11_0000;
    let _ = x | 0b11_1111 < 0b11_0000;
    let _ = x | 0b00_1111 >= 0b11_0000; //~ ineffective_bit_mask
    let _ = x | 0b11_0000 >= 0b11_0000;

    let _ = x ^ 0b00_0000 < 0b00_0000;
    let _ = x ^ 0b00_0001 < 0b00_0000;
    let _ = x ^ 0b00_0001 < 0b00_0001;
    let _ = x ^ 0b00_0010 < 0b00_0001;
    let _ = x ^ 0b00_0011 < 0b00_0001;
    let _ = x ^ 0b00_0000 < 0b00_0010; //~ ineffective_bit_mask
    let _ = x ^ 0b00_0001 < 0b00_0010; //~ ineffective_bit_mask
    let _ = x ^ 0b00_0010 < 0b00_0010;
    let _ = x ^ 0b00_0011 < 0b00_0010;
    let _ = x ^ 0b00_0000 < 0b00_0100; //~ ineffective_bit_mask
    let _ = x ^ 0b00_0001 < 0b00_0100; //~ ineffective_bit_mask
    let _ = x ^ 0b00_0010 < 0b00_0100; //~ ineffective_bit_mask
    let _ = x ^ 0b00_0011 < 0b00_0100; //~ ineffective_bit_mask
    let _ = x ^ 0b00_0100 < 0b00_0100;
    let _ = x ^ 0b00_0101 < 0b00_0100;
    let _ = x ^ 0b00_0001 < 0b00_1000; //~ ineffective_bit_mask
    let _ = x ^ 0b00_0010 < 0b00_1000; //~ ineffective_bit_mask
    let _ = x ^ 0b00_0011 < 0b00_1000; //~ ineffective_bit_mask
    let _ = x ^ 0b00_0100 < 0b00_1000; //~ ineffective_bit_mask
    let _ = x ^ 0b00_1000 < 0b00_1000;
    let _ = x ^ 0b00_1010 < 0b00_1000;
    let _ = x ^ 0b00_0001 < 0b01_0000; //~ ineffective_bit_mask
    let _ = x ^ 0b00_0010 < 0b01_0000; //~ ineffective_bit_mask
    let _ = x ^ 0b00_0011 < 0b01_0000; //~ ineffective_bit_mask
    let _ = x ^ 0b00_0100 < 0b01_0000; //~ ineffective_bit_mask
    let _ = x ^ 0b00_0101 < 0b01_0000; //~ ineffective_bit_mask
    let _ = x ^ 0b00_0110 < 0b01_0000; //~ ineffective_bit_mask
    let _ = x ^ 0b00_0111 < 0b01_0000; //~ ineffective_bit_mask
    let _ = x ^ 0b00_1000 < 0b01_0000; //~ ineffective_bit_mask
    let _ = x ^ 0b00_1001 < 0b01_0000; //~ ineffective_bit_mask
    let _ = x ^ 0b00_1111 < 0b11_0000; //~ ineffective_bit_mask
    let _ = x ^ 0b01_0000 < 0b11_0000;
    let _ = x ^ 0b01_1111 < 0b11_0000;
    let _ = x ^ 0b10_0000 < 0b11_0000;
    let _ = x ^ 0b11_1111 < 0b11_0000;
    let _ = x ^ 0b00_1111 >= 0b11_0000; //~ ineffective_bit_mask
    let _ = x ^ 0b11_0000 >= 0b11_0000;

    let _ = x | 0x7fff_ffff > 0xffff_ffff; //~ ineffective_bit_mask
    let _ = x | 0x8000_0000 > 0xffff_ffff; //~ ineffective_bit_mask
    let _ = x | 0xffff_ffff > 0xffff_ffff; //~ ineffective_bit_mask
    let _ = x ^ 0x7fff_ffff > 0xffff_ffff; //~ ineffective_bit_mask
    let _ = x ^ 0x8000_0000 > 0xffff_ffff; //~ ineffective_bit_mask
    let _ = x ^ 0xffff_ffff > 0xffff_ffff; //~ ineffective_bit_mask

    let _ = x | 0x7fff_ffff < 0x8000_0000; //~ ineffective_bit_mask
    let _ = x | 0x8000_0000 < 0x8000_0000;
    let _ = x ^ 0x7fff_ffff < 0x8000_0000; //~ ineffective_bit_mask
    let _ = x ^ 0x8000_0000 < 0x8000_0000;

    let _ = x | 0b00_0001 > 0b00_0100;
    let _ = x | 0b00_0010 > 0b00_0100;
    let _ = x | 0b00_0011 > 0b00_0100;
    let _ = x | 0b00_0100 > 0b00_0100;
    let _ = x | 0b00_0101 > 0b00_0100;

    let _ = x | 0b00_0001 < 0b00_1011;
    let _ = x | 0b00_0010 < 0b00_1011;
    let _ = x | 0b00_0011 < 0b00_1011;
    let _ = x | 0b00_0100 < 0b00_1011;
    let _ = x | 0b00_0101 < 0b00_1011;

    external! {
        let x = black_box(5u32);
        let _ = x | 0b0001 > 0b00_0011;
    }
    with_span! {
        sp
        let x = black_box(5u32);
        let _ = x | 0b0001 > 0b00_0011;
    }
}
