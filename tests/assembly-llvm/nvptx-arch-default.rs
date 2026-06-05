//@ assembly-output: target-linker-default
//@ compile-flags: --target nvptx64-nvidia-cuda --crate-type cdylib
//@ needs-llvm-components: nvptx
//@ ignore-backends: gcc
#![feature(no_core)]
#![no_core]

// Verify default arch with llvm-bitcode-linker.
// CHECK: .version 7.0
// CHECK: .target sm_70
// CHECK: .address_size 64
