// Derive macros can generate `macro_rules` items, regression test for issue #63651.

//@ check-pass
//@ proc-macro: gen-macro-rules.rs
//@ ignore-backends: gcc

extern crate gen_macro_rules as repro;

#[derive(repro::repro)]
pub struct S;

m!(); // OK

fn main() {}
