//@ add-core-stubs
//@ check-pass

#![feature(no_core, lang_items)]
#![no_core]
#![crate_type = "lib"]

extern crate minicore;
use minicore::*;

extern "C" {
    pub static A: u32;
}
