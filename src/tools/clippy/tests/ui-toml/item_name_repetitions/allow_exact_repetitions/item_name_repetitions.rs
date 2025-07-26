#![warn(clippy::module_name_repetitions)]
#![allow(dead_code)]

pub mod foo {
    // this line should produce a warning:
    pub fn foo() {}
    //~^ module_name_repetitions

    // but this line shouldn't
    pub fn to_foo() {}
}

fn main() {}
