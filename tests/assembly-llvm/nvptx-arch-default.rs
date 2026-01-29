//@ assembly-output: ptx-linker
//@ compile-flags: --target nvptx64-nvidia-cuda --crate-type cdylib -Ctarget-cpu=sm_30
//@ needs-llvm-components: nvptx
//@ ignore-backends: gcc
#![feature(no_core)]
#![no_core]

// Verify default target arch with ptx-linker.
// CHECK: .target sm_30
// CHECK: .address_size 64
