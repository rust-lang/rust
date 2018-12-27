// compile-pass
// skip-codegen
#![allow(unused)]
extern crate core;
use core as core_export;
use self::x::*;
mod x {}


fn main() {}
