// compile-flags: --crate-type=lib
#![feature(mut_restriction)]

pub struct Foo {
    pub mut x: u32, //~ ERROR expected `(`
}
