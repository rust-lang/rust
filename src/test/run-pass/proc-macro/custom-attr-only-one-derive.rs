// aux-build:custom-attr-only-one-derive.rs

#![feature(rust_2018_preview)]

#[macro_use]
extern crate custom_attr_only_one_derive;

#[derive(Bar, Foo)]
#[custom = "test"]
pub enum A {
    B,
    C,
}

fn main() {}
