// https://github.com/rust-lang/rust/issues/9249
//@ check-pass
#![allow(dead_code)]

static DATA:&'static [&'static str] = &["my string"];
fn main() { }
