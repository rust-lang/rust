#![feature(plugin)]
#![plugin(clippy)]
#![warn(stutter)]
#![allow(dead_code)]

mod foo {
    pub fn foo() {}
    pub fn foo_bar() {}
    pub fn bar_foo() {}
    pub struct FooCake {}
    pub enum CakeFoo {}
}

fn main() {}
