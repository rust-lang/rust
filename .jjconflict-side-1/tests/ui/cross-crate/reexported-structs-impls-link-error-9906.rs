// https://github.com/rust-lang/rust/issues/9906
//@ run-pass
//@ aux-build:aux-9906.rs

extern crate aux_9906 as testmod;

pub fn main() {
    testmod::foo();
    testmod::FooBar::new(1);
}
