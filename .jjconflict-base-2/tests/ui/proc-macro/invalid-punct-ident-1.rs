//@ proc-macro: invalid-punct-ident.rs
//@ needs-unwind proc macro panics to report errors

#[macro_use]
extern crate invalid_punct_ident;

invalid_punct!(); //~ ERROR proc macro panicked

fn main() {}
