// run-pass
#![allow(dead_code)]
// test for issue #30244

#[derive(Copy, Clone)]
struct Array {
    arr: [[u8; 256]; 4]
}

pub fn main() {}
