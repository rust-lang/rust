//@ run-pass

#![allow(dead_code)]

include!("auxiliary/issue-40469.rs");
fn f() { m!(); }

fn main() {}
