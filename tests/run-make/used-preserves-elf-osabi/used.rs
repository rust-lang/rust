#![feature(no_core)]
#![no_core]

extern crate minicore;

use minicore::*;

#[used]
pub static FOO: u8 = 42;
