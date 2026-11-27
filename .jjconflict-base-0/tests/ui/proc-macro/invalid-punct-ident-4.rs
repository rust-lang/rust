//@ proc-macro: invalid-punct-ident.rs
//@ check-pass

#[macro_use]
extern crate invalid_punct_ident;

lexer_failure!();

fn main() {}
