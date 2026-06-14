//@ assembly-output: emit-asm
//@ compile-flags: --target nvptx64-nvidia-cuda --crate-type rlib
//@ needs-llvm-components: nvptx
//@ ignore-backends: gcc
#![feature(no_core)]
#![no_core]

// Verify default arch without llvm-bitcode-linker involved.
// CHECK: .version 7.0
// CHECK: .target sm_70
// CHECK: .address_size 64
