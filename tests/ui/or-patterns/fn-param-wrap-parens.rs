// Test the suggestion to wrap an or-pattern as a function parameter in parens.

//@ run-rustfix

#![allow(warnings)]

fn main() {}

enum E { A, B }
use E::*;

#[cfg(FALSE)]
fn fun1(A | B: E) {} //~ ERROR top-level or-patterns are not allowed
