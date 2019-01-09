// run-pass
// Test that inconsistent bounds are used in well-formedness checks
#![feature(trivial_bounds)]

use std::fmt::Debug;

pub fn foo() where Vec<str>: Debug, str: Copy {
    let x = vec![*"1"];
    println!("{:?}", x);
}

fn main() {}
