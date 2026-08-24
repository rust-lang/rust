//@ no-prefer-dynamic
//@ compile-flags: --target nvptx64-nvidia-cuda
//@ needs-llvm-components: nvptx
//@ ignore-backends: gcc

#![feature(no_core)]
#![crate_type = "rlib"]
#![no_core]
