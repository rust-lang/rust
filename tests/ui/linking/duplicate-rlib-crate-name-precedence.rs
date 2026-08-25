//! Regression test for <https://github.com/rust-lang/rust/issues/18913>.

//@ run-pass
//@ aux-build:duplicate-rlib-crate-name-precedence-1.rs
//@ aux-build:duplicate-rlib-crate-name-precedence-2.rs

extern crate foo;

fn main() {
    assert_eq!(foo::foo(), 1);
}
