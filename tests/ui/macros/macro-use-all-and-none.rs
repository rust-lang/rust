//@ check-pass
//@ aux-build:two_macros-rpass.rs
//@ reference: macro.decl.scope.macro_use.duplicates

#![warn(unused_attributes)]

#[macro_use]
#[macro_use()] //~ WARNING unused attribute
extern crate two_macros_rpass;

pub fn main() {
    macro_one!();
    macro_two!();
}
