// Checks that the gpu-kernel calling convention correctly translates to LLVM calling conventions.

//@ add-core-stubs
//@ revisions: amdgpu nvptx
//@ [amdgpu] compile-flags: --crate-type=rlib --target=amdgcn-amd-amdhsa -Ctarget-cpu=gfx900
//@ [amdgpu] needs-llvm-components: amdgpu
//@ [nvptx] compile-flags: --crate-type=rlib --target=nvptx64-nvidia-cuda
//@ [nvptx] needs-llvm-components: nvptx
#![feature(no_core, lang_items, abi_gpu_kernel)]
#![no_core]

extern crate minicore;
use minicore::*;

// amdgpu: define amdgpu_kernel void @fun(i32
// nvptx: define ptx_kernel void @fun(i32
#[no_mangle]
pub extern "gpu-kernel" fn fun(_: i32) {}
