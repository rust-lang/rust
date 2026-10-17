//@ run-pass
#![allow(dead_code)]

trait Foo {
    fn dummy(&self) { }
}

fn foo<'a, 'b, 'c:'a+'b, 'd>() {
}

pub fn main() { }
