//@ assembly-output: emit-asm
//@ compile-flags: --crate-type rlib
//@ only-nvptx64

#![no_std]

// Verify default arch without llvm-bitcode-linker involved.
// CHECK: .version 7.0
// CHECK: .target sm_70
// CHECK: .address_size 64
