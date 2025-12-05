#![crate_type = "dylib"]
extern crate both;

use std::mem;

pub fn addr() -> usize {
    unsafe { mem::transmute(&both::foo) }
}
