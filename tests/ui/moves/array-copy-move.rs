//! regression test for issue https://github.com/rust-lang/rust/issues/16783
//@ run-pass
#![allow(unused_variables)]

pub fn main() {
    let x = [1, 2, 3];
    let y = x;
}
