//@ add-minicore
//@ check-pass

#![feature(no_core)]
#![no_core]
#![crate_type = "lib"]

extern crate minicore;
use minicore::*;

extern "C" {
    pub static A: u32;
}
