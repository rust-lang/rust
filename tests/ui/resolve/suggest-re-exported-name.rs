//@ run-rustfix
use std::fmt::Debug as OtherDebug;

pub struct Debug;

pub fn foo() -> impl Debug { //~ ERROR: expected trait, found struct `Debug`
    ""
}

pub fn bar() -> impl crate::Debug { //~ ERROR: expected trait, found struct `crate::Debug`
    ""
}

pub fn baz<T: Debug>() {} //~ ERROR: expected trait, found struct `Debug`

fn main() {}
