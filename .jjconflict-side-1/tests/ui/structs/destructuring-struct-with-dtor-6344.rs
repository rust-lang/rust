// https://github.com/rust-lang/rust/issues/6344
//@ run-pass
#![allow(non_shorthand_field_patterns)]

struct A { x: usize }

impl Drop for A {
    fn drop(&mut self) {}
}

pub fn main() {
    let a = A { x: 0 };

    let A { x: ref x } = a;
    println!("{}", x)
}
