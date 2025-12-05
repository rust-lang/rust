//@ run-pass
#![allow(dead_code)]
// regression test for issue 4875


pub struct Foo<T> {
    data: T,
}

fn foo<T>(Foo{..}: Foo<T>) {
}

pub fn main() {
}
