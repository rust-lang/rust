#![feature(prelude_import)]
#![no_std]
#![feature(export)]
#![crate_type = "dylib"]
#[prelude_import]
use ::std::prelude::rust_2015::*;
#[macro_use]
extern crate std;

// interface file is broken:
#[export]
pub extern "C" fn foo()
