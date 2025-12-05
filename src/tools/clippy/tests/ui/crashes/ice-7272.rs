//@ check-pass
//@aux-build:ice-7272-aux.rs

#![allow(clippy::no_effect)]

extern crate ice_7272_aux;

use ice_7272_aux::*;

pub fn main() {
    || WARNING!("Style changed!");
    || "}{";
}
