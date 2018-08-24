#![feature(rustc_attrs)]
#![allow(unused)]

extern crate core;
use core as core_export;
use self::x::*;
mod x {}

#[rustc_error]
fn main() {} //~ ERROR compilation successful
