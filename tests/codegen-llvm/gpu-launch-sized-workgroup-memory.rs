// Checks that the GPU intrinsic to get launch-sized workgroup memory works.

//@ revisions: amdgpu nvptx
//@ compile-flags: --crate-type=rlib
//
//@ [amdgpu] compile-flags: --target amdgcn-amd-amdhsa -Ctarget-cpu=gfx900
//@ [amdgpu] needs-llvm-components: amdgpu
//@ [nvptx] compile-flags: --target nvptx64-nvidia-cuda
//@ [nvptx] needs-llvm-components: nvptx
//@ add-minicore
#![feature(intrinsics, no_core, rustc_attrs)]
#![no_core]

extern crate minicore;

#[rustc_intrinsic]
#[rustc_nounwind]
fn gpu_launch_sized_workgroup_mem<T>() -> *mut T;

// amdgpu-DAG: @[[SMALL:[^ ]+]] = external addrspace(3) global [0 x i8], align 4
// amdgpu-DAG: @[[BIG:[^ ]+]] = external addrspace(3) global [0 x i8], align 8
// amdgpu: ret { ptr, ptr } { ptr addrspacecast (ptr addrspace(3) @[[SMALL]] to ptr), ptr addrspacecast (ptr addrspace(3) @[[BIG]] to ptr) }

// nvptx: @[[BIG:[^ ]+]] = external addrspace(3) global [0 x i8], align 8
// nvptx: ret { ptr, ptr } { ptr addrspacecast (ptr addrspace(3) @[[BIG]] to ptr), ptr addrspacecast (ptr addrspace(3) @[[BIG]] to ptr) }
#[unsafe(no_mangle)]
pub fn fun() -> (*mut i32, *mut f64) {
    let small = gpu_launch_sized_workgroup_mem::<i32>();
    let big = gpu_launch_sized_workgroup_mem::<f64>(); // Increase alignment to 8
    (small, big)
}
