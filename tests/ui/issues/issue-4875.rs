//@ run-pass
#![allow(dead_code)]
#![allow(unused_unconstructable_pub_structs)]
// regression test for issue 4875


pub struct Foo<T> {
    data: T,
}

fn foo<T>(Foo{..}: Foo<T>) {
}

pub fn main() {
}
