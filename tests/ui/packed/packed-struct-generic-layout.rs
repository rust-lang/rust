// run-pass
#![allow(dead_code)]
#![allow(overflowing_literals)]


use std::mem;

#[repr(packed)]
struct S<T, S> {
    a: T,
    b: u8,
    c: S
}

pub fn main() {
    unsafe {
        let s = S { a: 0xff_ff_ff_ffu32, b: 1, c: 0xaa_aa_aa_aa as i32 };
        let transd : [u8; 9] = mem::transmute(s);
        // Don't worry about endianness, the numbers are palindromic.
        assert_eq!(transd,
                   [0xff, 0xff, 0xff, 0xff,
                    1,
                    0xaa, 0xaa, 0xaa, 0xaa]);


        let s = S { a: 1u8, b: 2u8, c: 0b10000001_10000001 as i16};
        let transd : [u8; 4] = mem::transmute(s);
        // Again, no endianness problems.
        assert_eq!(transd,
                   [1, 2, 0b10000001, 0b10000001]);
    }
}
