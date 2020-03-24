// aux-build:invalid-punct-ident.rs

#[macro_use]
extern crate invalid_punct_ident;

lexer_failure!();
//~^ ERROR proc macro panicked
//~| ERROR unexpected closing delimiter: `)`

fn main() {
    let _recovery_witness: () = 0; //~ ERROR mismatched types
}
