//@ assembly-output: ptx-linker
//@ compile-flags: --crate-type cdylib -Z unstable-options -Clinker-flavor=llbc
//@ only-nvptx64

#![no_std]

//@ aux-build: breakpoint-panic-handler.rs
extern crate breakpoint_panic_handler;

// Verify default target arch with ptx-linker.
// CHECK: .target sm_30
// CHECK: .address_size 64
