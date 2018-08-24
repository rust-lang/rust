#![feature(pub_restricted)]

struct Bar(pub(()));

struct Foo {
    pub(crate) () foo: usize, //~ ERROR expected identifier
}

fn main() {}
