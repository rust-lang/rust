// run-rustfix
// compile-flags: --crate-type=lib

#![feature(mut_restriction)]

pub mod a {
    pub struct Foo {
        mut(crate::a) _foo: u8, //~ ERROR incorrect mut restriction
    }
}
