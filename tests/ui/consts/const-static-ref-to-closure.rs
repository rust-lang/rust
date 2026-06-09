//! regression test for <https://github.com/rust-lang/rust/issues/25180>
//@ check-pass
#![allow(dead_code)]

const X: &'static dyn Fn() = &|| println!("ICE here");

fn main() {}
