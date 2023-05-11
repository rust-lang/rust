// assembly-output: emit-asm
// compile-flags: --crate-type rlib
// only-nvptx64
// ignore-nvptx64

#![no_std]

// Verify default arch without ptx-linker involved.
// CHECK: .target sm_30
// CHECK: .address_size 64
