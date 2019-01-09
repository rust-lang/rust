// run-pass

#![allow(dead_code)]

#[derive]   //~ WARNING empty trait list in `derive`
struct Foo;

#[derive()] //~ WARNING empty trait list in `derive`
struct Bar;

pub fn main() {}
