//! Regression test for <https://github.com/rust-lang/rust/issues/18988>.

//@ check-pass
#![allow(dead_code)]
pub trait Foo : Send { }

pub struct MyFoo {
    children: Vec<Box<dyn Foo>>,
}

impl Foo for MyFoo { }

pub fn main() { }
