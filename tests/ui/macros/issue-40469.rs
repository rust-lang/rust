// run-pass
// ignore-pretty issue #37195

#![allow(dead_code)]

include!("auxiliary/issue-40469.rs");
fn f() { m!(); }

fn main() {}
