//@ run-pass
//@ aux-build:inline_dtor.rs


extern crate inline_dtor;

pub fn main() {
    let _x = inline_dtor::Foo;
}
