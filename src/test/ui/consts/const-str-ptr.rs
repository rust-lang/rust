// run-pass
#![allow(unused_imports)]
use std::{str, string};

const A: [u8; 2] = ['h' as u8, 'i' as u8];
const B: &'static [u8; 2] = &A;
const C: *const u8 = B as *const u8;

pub fn main() {
    unsafe {
        let foo = &A as *const u8;
        assert_eq!(foo, C);
        assert_eq!(str::from_utf8_unchecked(&A), "hi");
        assert_eq!(*C, A[0]);
        assert_eq!(*(&B[0] as *const u8), A[0]);
    }
}
