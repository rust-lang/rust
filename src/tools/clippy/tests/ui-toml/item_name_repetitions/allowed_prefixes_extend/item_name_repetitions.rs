#![warn(clippy::module_name_repetitions)]
#![allow(dead_code)]

pub mod foo {
    // #12544 - shouldn't warn if item name consists only of an allowed prefix and a module name.
    // In this test, allowed prefixes are configured to be all of the default prefixes and ["bar"].

    // this line should produce a warning:
    pub fn something_foo() {}
    //~^ module_name_repetitions

    // but none of the following should:
    pub fn bar_foo() {}
    pub fn to_foo() {}
    pub fn as_foo() {}
    pub fn into_foo() {}
    pub fn from_foo() {}
    pub fn try_into_foo() {}
    pub fn try_from_foo() {}
}

fn main() {}
