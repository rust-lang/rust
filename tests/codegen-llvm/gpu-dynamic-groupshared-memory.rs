// Checks that the GPU dynamic group-shared memory intrinsic works.

//@ revisions: amdgpu nvptx
//@ compile-flags: --crate-type=rlib
//
//@ [amdgpu] compile-flags: --target amdgcn-amd-amdhsa -Ctarget-cpu=gfx900
//@ [amdgpu] needs-llvm-components: amdgpu
//@ [nvptx] compile-flags: --target nvptx64-nvidia-cuda
//@ [nvptx] needs-llvm-components: nvptx
//@ add-core-stubs
#![feature(intrinsics, no_core, rustc_attrs)]
#![no_core]

extern crate minicore;

#[rustc_intrinsic]
#[rustc_nounwind]
fn gpu_dynamic_groupshared_mem<T>() -> *mut T;

// CHECK: @gpu_dynamic_groupshared_mem = external addrspace(3) global [0 x i8], align 8
// CHECK: ret ptr addrspacecast (ptr addrspace(3) @gpu_dynamic_groupshared_mem to ptr)
#[unsafe(no_mangle)]
pub fn fun() -> *mut i32 {
    let res = gpu_dynamic_groupshared_mem::<i32>();
    gpu_dynamic_groupshared_mem::<f64>(); // Increase alignment to 8
    res
}
