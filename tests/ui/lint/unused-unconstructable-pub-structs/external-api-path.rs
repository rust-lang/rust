//@ check-pass

#![deny(unused_unconstructable_pub_structs)]

pub struct Foo(i32);

pub fn foo(x: Foo) {
    let _ = x;
}

fn main() {}
