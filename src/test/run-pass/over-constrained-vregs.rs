#![allow(unused_must_use)]
// Regression test for issue #152.
pub fn main() {
    let mut b: usize = 1_usize;
    while b < std::mem::size_of::<usize>() {
        0_usize << b;
        b <<= 1_usize;
        println!("{}", b);
    }
}
