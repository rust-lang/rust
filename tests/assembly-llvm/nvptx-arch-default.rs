//@ assembly-output: ptx-linker
//@ compile-flags: --crate-type cdylib
//@ only-nvptx64

#![no_std]

//@ aux-build: breakpoint-panic-handler.rs
extern crate breakpoint_panic_handler;

// Verify default target arch with ptx-linker.
// CHECK: .version 7.0
// CHECK: .target sm_70
// CHECK: .address_size 64
