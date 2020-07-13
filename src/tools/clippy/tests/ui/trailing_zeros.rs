#![allow(unused_parens)]

fn main() {
    let x: i32 = 42;
    let _ = (x & 0b1111 == 0); // suggest trailing_zeros
    let _ = x & 0b1_1111 == 0; // suggest trailing_zeros
    let _ = x & 0b1_1010 == 0; // do not lint
    let _ = x & 1 == 0; // do not lint
}
