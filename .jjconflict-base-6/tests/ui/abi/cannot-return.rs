//@ add-minicore
//@ ignore-backends: gcc
//@ edition: 2024
//@ revisions: amdgpu nvptx
//
//@ [amdgpu] needs-llvm-components: amdgpu
//@ [amdgpu] compile-flags: --target amdgcn-amd-amdhsa -Ctarget-cpu=gfx900 --crate-type=rlib
//@ [nvptx] needs-llvm-components: nvptx
//@ [nvptx] compile-flags: --target nvptx64-nvidia-cuda --crate-type=rlib
#![no_core]
#![feature(no_core, abi_gpu_kernel)]

extern crate minicore;
use minicore::*;

#[unsafe(no_mangle)]
extern "gpu-kernel" fn ret_i32() -> i32 { 0 }
//~^ ERROR invalid signature for `extern "gpu-kernel"` function
