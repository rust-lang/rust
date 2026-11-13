//@ edition: 2018
#![feature(unqualified_local_imports)]
#![deny(unqualified_local_imports)]

mod localmod {
    pub struct S;
    pub struct T;
}

// Not a local import, so no lint.
use std::cell::Cell;

// Implicitly local import, gets lint.
use localmod::S; //~ERROR: unqualified

// Explicitly local import, no lint.
use self::localmod::T;

macro_rules! mymacro {
    ($cond:expr) => {
        if !$cond {
            continue;
        }
    };
}
// Macro import: no lint, as there is no other way to write it.
pub(crate) use mymacro;

#[allow(unused)]
enum LocalEnum {
    VarA,
    VarB,
}

fn main() {
    // Import in a function, no lint.
    use LocalEnum::*;
}
