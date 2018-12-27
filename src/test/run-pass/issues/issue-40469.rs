// run-pass
// ignore-pretty issue #37195

#![allow(dead_code)]

include!("auxiliary/issue_40469.rs");
fn f() { m!(); }

fn main() {}
