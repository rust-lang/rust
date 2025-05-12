//@ run-pass
#![allow(dead_code)]
use std::mem;

#[repr(packed)]
struct S4 {
    a: u8,
    b: [u8; 3],
}

#[repr(packed)]
struct S5 {
    a: u8,
    b: u32
}

pub fn main() {
    unsafe {
        let s4 = S4 { a: 1, b: [2,3,4] };
        let transd : [u8; 4] = mem::transmute(s4);
        assert_eq!(transd, [1, 2, 3, 4]);

        let s5 = S5 { a: 1, b: 0xff_00_00_ff };
        let transd : [u8; 5] = mem::transmute(s5);
        // Don't worry about endianness, the u32 is palindromic.
        assert_eq!(transd, [1, 0xff, 0, 0, 0xff]);
    }
}
