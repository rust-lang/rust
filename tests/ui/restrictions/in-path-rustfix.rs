// run-rustfix

#![crate_type = "lib"]
#![feature(mut_restriction)]

pub mod a {
    pub struct Foo {
        mut(crate::a) _foo: u8, //~ ERROR incorrect mut restriction
    }
}
