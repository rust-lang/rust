// assembly-output: ptx-linker
// compile-flags: --crate-type cdylib -C link-arg=--arch=sm_60
// only-nvptx64
// ignore-nvptx64

#![no_std]

// aux-build: breakpoint-panic-handler.rs
extern crate breakpoint_panic_handler;

// Verify target arch override via `link-arg`.
// CHECK: .target sm_60
// CHECK: .address_size 64
