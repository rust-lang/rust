//@ compile-flags:-C panic=abort
//@ no-prefer-dynamic

#![crate_type = "rlib"]
#![no_std]

extern crate panic_runtime_abort;
