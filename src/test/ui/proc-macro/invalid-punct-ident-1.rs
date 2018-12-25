// aux-build:invalid-punct-ident.rs

#[macro_use]
extern crate invalid_punct_ident;

invalid_punct!(); //~ ERROR proc macro panicked
