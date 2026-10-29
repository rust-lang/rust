//@aux-build:proc_macros.rs

#![warn(clippy::bad_bit_mask)]
#![expect(clippy::erasing_op, clippy::identity_op)]

use core::hint::black_box;
use core::ops::BitAnd;
use proc_macros::{external, with_span};

fn main() {
    let x = black_box(5u32);

    let _ = x & 0b0000 == 0b0000; //~ bad_bit_mask
    let _ = x & 0b0000 == 0b0001; //~ bad_bit_mask
    let _ = x & 0b0001 == 0b0000;
    let _ = x & 0b0001 == 0b0001;
    let _ = x & 0b0010 == 0b0001; //~ bad_bit_mask
    let _ = x & 0b0001 == 0b0010; //~ bad_bit_mask
    let _ = x & 0b0011 == 0b0001;
    let _ = x & 0b0011 == 0b0010;
    let _ = x & 0b0011 == 0b0011;
    let _ = x & 0b0011 == 0b0100; //~ bad_bit_mask
    let _ = x & 0b0100 == 0b0100;
    let _ = x & 0b0110 == 0b0111; //~ bad_bit_mask
    let _ = x & 0b0011 == 0b0101; //~ bad_bit_mask
    let _ = x & 0b1111 == 0b1110;
    let _ = x & 0b1111 == 0b1010;

    let _ = x | 0b0000 == 0b0000;
    let _ = x | 0b0000 == 0b0001;
    let _ = x | 0b0001 == 0b0000; //~ bad_bit_mask
    let _ = x | 0b0001 == 0b0001;
    let _ = x | 0b0010 == 0b0001; //~ bad_bit_mask
    let _ = x | 0b0001 == 0b0010; //~ bad_bit_mask
    let _ = x | 0b0011 == 0b0001; //~ bad_bit_mask
    let _ = x | 0b0011 == 0b0010; //~ bad_bit_mask
    let _ = x | 0b0011 == 0b0011;
    let _ = x | 0b0011 == 0b0100; //~ bad_bit_mask
    let _ = x | 0b0100 == 0b0100;
    let _ = x | 0b0110 == 0b0111;
    let _ = x | 0b0011 == 0b0101; //~ bad_bit_mask
    let _ = x | 0b1111 == 0b1110; //~ bad_bit_mask
    let _ = x | 0b1111 == 0b1010; //~ bad_bit_mask

    let _ = x & 0b0000 != 0b0000; //~ bad_bit_mask
    let _ = x & 0b0000 != 0b0001; //~ bad_bit_mask
    let _ = x & 0b0001 != 0b0000;
    let _ = x & 0b0001 != 0b0001;
    let _ = x & 0b0010 != 0b0001; //~ bad_bit_mask
    let _ = x & 0b0001 != 0b0010; //~ bad_bit_mask
    let _ = x & 0b0011 != 0b0001;
    let _ = x & 0b0011 != 0b0010;
    let _ = x & 0b0011 != 0b0011;
    let _ = x & 0b0011 != 0b0100; //~ bad_bit_mask
    let _ = x & 0b0100 != 0b0100;
    let _ = x & 0b0110 != 0b0111; //~ bad_bit_mask
    let _ = x & 0b0011 != 0b0101; //~ bad_bit_mask
    let _ = x & 0b1111 != 0b1110;
    let _ = x & 0b1111 != 0b1010;

    let _ = x | 0b0000 != 0b0000;
    let _ = x | 0b0000 != 0b0001;
    let _ = x | 0b0001 != 0b0000; //~ bad_bit_mask
    let _ = x | 0b0001 != 0b0001;
    let _ = x | 0b0010 != 0b0001; //~ bad_bit_mask
    let _ = x | 0b0001 != 0b0010; //~ bad_bit_mask
    let _ = x | 0b0011 != 0b0001; //~ bad_bit_mask
    let _ = x | 0b0011 != 0b0010; //~ bad_bit_mask
    let _ = x | 0b0011 != 0b0011;
    let _ = x | 0b0011 != 0b0100; //~ bad_bit_mask
    let _ = x | 0b0100 != 0b0100;
    let _ = x | 0b0110 != 0b0111;
    let _ = x | 0b0011 != 0b0101; //~ bad_bit_mask
    let _ = x | 0b1111 != 0b1110; //~ bad_bit_mask
    let _ = x | 0b1111 != 0b1010; //~ bad_bit_mask

    let _ = 0b0010 & x == 0b0001; //~ bad_bit_mask
    let _ = 0b0001 == x & 0b0010; //~ bad_bit_mask
    let _ = 0b0001 == 0b0010 & x; //~ bad_bit_mask

    let _ = x & (0b0100 | 0b0010) == (0b0111 ^ 0b1000); //~ bad_bit_mask

    external! {
        let x = black_box(5u32);
        let _ = x & 0b0010 == 0b0001;
    }
    with_span! {
        sp
        let x = black_box(5u32);
        let _ = x & 0b0010 == 0b0001;
    }

    {
        const C: i32 = 0b0011;

        let x = black_box(5i32);
        let _ = x & C == 0b0011;
        let _ = x & C == 0b0100; //~ bad_bit_mask
        let _ = x & 0b0001 == C; //~ bad_bit_mask
    }

    {
        // Bits shifted out.
        let _ = black_box(1u8) & 0xf0 == 0x11 << 4;
        let _ = black_box(1i8) & 0x70 == 0x11 << 4;
        let _ = black_box(1u16) & 0xf000 == 0x11 << 12;
        let _ = black_box(1i16) & 0x7000 == 0x11 << 12;
    }

    {
        struct S(u32);
        impl BitAnd<u32> for S {
            type Output = Self;
            fn bitand(self, _: u32) -> Self {
                self
            }
        }
        impl PartialEq<u32> for S {
            fn eq(&self, _: &u32) -> bool {
                true
            }
        }
        let _ = black_box(S(0)) & 0x1 != 0;
    }
}
