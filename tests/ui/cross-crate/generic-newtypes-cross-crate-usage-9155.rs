// https://github.com/rust-lang/rust/issues/9155
//@ run-pass
//@ aux-build:aux-9155.rs

extern crate aux_9155;

struct Baz;

pub fn main() {
    aux_9155::Foo::new(Baz);
}
