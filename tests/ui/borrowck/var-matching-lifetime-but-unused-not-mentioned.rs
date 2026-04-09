//! Regression test for <https://github.com/rust-lang/rust/issues/113121>.

#![allow(unused_variables)]

fn consume<T: 'static>(_: T) {}

fn foo<'a>(
    used_arg: &'a u8,
    unused_arg: &'static u16, // Unused in closure. Must not appear in error.
) {
    let unused_var: &'static u32 = &42; // Unused in closure. Must not appear in error.

    let c = move || used_arg;
    consume(c); //~ ERROR: borrowed data escapes outside of function
}

fn main() {}
