//@ check-pass

#![allow(clippy::disallowed_names, clippy::uninlined_format_args)]

pub fn foo(bar: *const u8) {
    println!("{:#p}", bar);
}

// Regression test for https://github.com/rust-lang/rust-clippy/issues/4917
/// <foo
struct A;

fn main() {}
