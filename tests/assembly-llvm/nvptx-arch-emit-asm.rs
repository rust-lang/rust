//@ assembly-output: emit-asm
//@ compile-flags: --target nvptx64-nvidia-cuda --crate-type rlib -Ctarget-cpu=sm_30
//@ needs-llvm-components: nvptx
//@ ignore-backends: gcc
#![feature(no_core)]
#![no_core]

// Verify default arch without ptx-linker involved.
// CHECK: .target sm_30
// CHECK: .address_size 64
