#![feature(plugin)]
#![plugin(clippy)]
#![deny(stutter)]
#![allow(dead_code)]

mod foo {
    pub fn foo() {}
    pub fn foo_bar() {} //~ ERROR: item name starts with its containing module's name
    pub fn bar_foo() {} //~ ERROR: item name ends with its containing module's name
    pub struct FooCake {} //~ ERROR: item name starts with its containing module's name
    pub enum CakeFoo {} //~ ERROR: item name ends with its containing module's name
}

fn main() {}
