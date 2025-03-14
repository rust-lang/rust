//@ proc-macro: issue-66286.rs

// Regression test for #66286.

extern crate issue_66286;

#[issue_66286::vec_ice]
pub extern "C" fn foo(_: Vec(u32)) -> u32 {
    //~^ ERROR: parenthesized type parameters may only be used with a `Fn` trait
    0
}

fn main() {}
