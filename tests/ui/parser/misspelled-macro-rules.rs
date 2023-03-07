// Regression test for issue #91227.

// run-rustfix

#![allow(unused_macros)]

marco_rules! thing {
//~^ ERROR: expected one of
//~| HELP: perhaps you meant to define a macro
    () => {}
}

fn main() {}
