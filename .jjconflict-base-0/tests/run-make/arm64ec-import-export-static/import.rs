#![crate_type = "cdylib"]
#![allow(internal_features)]
#![feature(no_core)]
#![no_std]
#![no_core]

extern crate export;

#[no_mangle]
pub extern "C" fn func() -> i32 {
    export::VALUE
}
