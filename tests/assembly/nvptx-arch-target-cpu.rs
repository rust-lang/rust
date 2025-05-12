//@ assembly-output: ptx-linker
//@ compile-flags: --crate-type cdylib -C target-cpu=sm_50 -Z unstable-options -Clinker-flavor=llbc
//@ only-nvptx64

#![no_std]

//@ aux-build: breakpoint-panic-handler.rs
extern crate breakpoint_panic_handler;

// Verify target arch override via `target-cpu`.
// CHECK: .target sm_50
// CHECK: .address_size 64
