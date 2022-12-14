// run-pass

#![allow(unused_imports)]
#![feature(rustc_private)]

extern crate libc;
use libc::c_void;

pub fn main() {
    println!("Hello world!");
}
