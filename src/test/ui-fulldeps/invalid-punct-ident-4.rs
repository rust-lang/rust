// aux-build:invalid-punct-ident.rs

#[macro_use]
extern crate invalid_punct_ident;

lexer_failure!(); //~ ERROR proc macro panicked
                  //~| ERROR unexpected close delimiter: `)`
