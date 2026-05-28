//@ proc-macro: external-macro-use.rs

extern crate external_macro_use;

#[unsafe(external_macro_use::a)]
//~^ ERROR unnecessary `unsafe` on safe attribute
fn f() {}

fn main() {}
