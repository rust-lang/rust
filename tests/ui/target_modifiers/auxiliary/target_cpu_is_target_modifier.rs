//@ no-prefer-dynamic
//@ compile-flags: --target nvptx64-nvidia-cuda -Ctarget-cpu=sm_60
//@ needs-llvm-components: nvptx

#![feature(no_core)]
#![crate_type = "rlib"]
#![no_core]
