#![warn(clippy::module_name_repetitions)]
#![allow(dead_code)]

pub mod foo {
    // #12544 - shouldn't warn if item name consists only of an allowed prefix and a module name.
    // In this test, allowed prefixes are configured to be ["bar"].

    // this line should produce a warning:
    pub fn to_foo() {}
    //~^ module_name_repetitions

    // but this line shouldn't
    pub fn bar_foo() {}
}

fn main() {}
