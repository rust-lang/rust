#![allow(clippy::blacklisted_name)]

pub fn foo(bar: *const u8) {
    println!("{:#p}", bar);
}

// Regression test for https://github.com/rust-lang/rust-clippy/issues/4917
/// <foo
struct A {}

fn main() {}
