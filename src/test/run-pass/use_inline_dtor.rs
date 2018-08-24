// aux-build:inline_dtor.rs

// pretty-expanded FIXME #23616

extern crate inline_dtor;

pub fn main() {
    let _x = inline_dtor::Foo;
}
