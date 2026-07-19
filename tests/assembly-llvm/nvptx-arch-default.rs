//@ assembly-output: emit-asm
//@ compile-flags: --target nvptx64-nvidia-cuda --crate-type cdylib
//@ needs-llvm-components: nvptx
#![feature(no_core)]
#![no_core]

// Verify default arch with llvm-bitcode-linker.
// CHECK: .version 7.0
// CHECK: .target sm_70
// CHECK: .address_size 64
