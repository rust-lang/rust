#![allow(internal_features)]
#![feature(no_core, lang_items)]
#![no_core]
#![crate_type = "cdylib"]

extern crate lib;

#[unsafe(no_mangle)]
pub fn bar() {
    lib::foo();
}
