//@ assembly-output: emit-asm
//@ compile-flags: --target nvptx64-nvidia-cuda --crate-type cdylib -C target-cpu=sm_87
//@ needs-llvm-components: nvptx
//@ ignore-backends: gcc
#![feature(no_core)]
#![no_core]

// Verify target arch override via `target-cpu`.
// CHECK: .target sm_87
// CHECK: .address_size 64
