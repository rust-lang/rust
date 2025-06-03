//@ check-pass
//@ proc-macro: external-macro-use.rs

// issue#140255

extern crate external_macro_use as macro_use;

#[macro_use::a]
fn f() {}

fn main() {}
