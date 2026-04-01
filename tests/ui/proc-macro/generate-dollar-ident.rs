// Proc macros can generate token sequence `$ IDENT`
// without it being recognized as an unknown macro variable.

//@ check-pass
//@ proc-macro: generate-dollar-ident.rs

extern crate generate_dollar_ident;
use generate_dollar_ident::*;

macro_rules! black_hole {
    ($($tt:tt)*) => {};
}

black_hole!($var);

dollar_ident!(black_hole);

fn main() {}
