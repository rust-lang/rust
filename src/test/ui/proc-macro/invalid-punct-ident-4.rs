// aux-build:invalid-punct-ident.rs

// We use `main` not found below as a witness for error recovery in proc macro expansion.

#[macro_use] //~ ERROR `main` function not found
extern crate invalid_punct_ident;

lexer_failure!();
//~^ ERROR proc macro panicked
//~| ERROR unexpected closing delimiter: `)`
