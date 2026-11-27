// https://github.com/rust-lang/rust/issues/9188
//@ run-pass
//@ aux-build:aux-9188.rs

extern crate aux_9188 as lib;

pub fn main() {
    let a = lib::bar();
    let b = lib::foo::<isize>();
    assert_eq!(*a, *b);
}
