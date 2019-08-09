// assembly-output: ptx-linker
// compile-flags: --crate-type cdylib
// only-nvptx64
// ignore-nvptx64

#![no_std]

// aux-build: breakpoint-panic-handler.rs
extern crate breakpoint_panic_handler;

// Verify default target arch with ptx-linker.
// CHECK: .target sm_30
// CHECK: .address_size 64
