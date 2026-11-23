//! Regression test for <https://github.com/rust-lang/rust/issues/157152>.
//@ check-pass
//@ compile-flags: -Znext-solver=globally
#![feature(gca_const_items, gca_min_const_items)]

struct FieldName<const N: usize>([u8; N]);

impl FieldName<0> {
    const fn len(name: &str) -> usize {
        name.len()
    }
}

fn takes(_: FieldName<{ FieldName::len("id") }>) {}

fn main() {
    takes(FieldName([0; 2]));
}
