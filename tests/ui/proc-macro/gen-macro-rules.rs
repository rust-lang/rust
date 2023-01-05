// Derive macros can generate `macro_rules` items, regression test for issue #63651.

// check-pass
// aux-build:gen-macro-rules.rs

extern crate gen_macro_rules as repro;

#[derive(repro::repro)]
pub struct S;

m!(); // OK

fn main() {}
