//@ compile-flags: -Z unstable-options
//@ ignore-stage1

#![feature(rustc_private)]
#![deny(rustc::direct_use_of_rustc_type_ir)]

extern crate rustc_type_ir;

use rustc_type_ir::*;
//~^ ERROR: do not use `rustc_type_ir` unless you are implementing type system internals


fn main() {}
