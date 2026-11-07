//@ check-pass
//@aux-build:ice-7272-aux.rs

#![expect(clippy::no_effect)]

extern crate ice_7272_aux;

use ice_7272_aux::*;

pub fn main() {
    || WARNING!("Style changed!");
    || "}{";
}
