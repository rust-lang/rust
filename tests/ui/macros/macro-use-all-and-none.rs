//@ run-pass
//@ aux-build:two_macros-rpass.rs

#![warn(unused_attributes)]

#[macro_use]
#[macro_use()] //~ WARNING unused attribute
extern crate two_macros_rpass;

pub fn main() {
    macro_one!();
    macro_two!();
}
