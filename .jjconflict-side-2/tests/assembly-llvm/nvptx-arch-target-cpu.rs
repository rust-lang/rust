//@ assembly-output: emit-asm
//@ compile-flags: --crate-type cdylib -C target-cpu=sm_87
//@ only-nvptx64

#![feature(no_core)]
#![no_core]

// Verify target arch override via `target-cpu`.
// CHECK: .target sm_87
// CHECK: .address_size 64
