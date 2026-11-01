//@ run-rustfix
// https://github.com/rust-lang/rust/issues/164044
#![allow(dead_code)]

struct A { b: Vec<u8>, c: usize }

fn main() {
    let _ = A(vec![], 2);
    //~^ ERROR cannot find function, tuple struct or tuple variant
}
