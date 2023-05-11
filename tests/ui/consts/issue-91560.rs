// Regression test for issue #91560.

// run-rustfix

#![allow(unused,non_upper_case_globals)]

fn foo() {
    let mut length: usize = 2;
    //~^ HELP: consider using `const`
    let arr = [0; length];
    //~^ ERROR: attempt to use a non-constant value in a constant [E0435]
}

fn bar() {
    let   length: usize = 2;
    //~^ HELP: consider using `const`
    let arr = [0; length];
    //~^ ERROR: attempt to use a non-constant value in a constant [E0435]
}

fn main() {}
