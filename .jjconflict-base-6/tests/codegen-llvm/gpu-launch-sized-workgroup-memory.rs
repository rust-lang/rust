// Checks that the GPU intrinsic to get launch-sized workgroup memory works
// and correctly aligns the `external addrspace(...) global`s over multiple calls.

//@ revisions: amdgpu nvptx-pre-llvm-23 nvptx-post-llvm-23
//@ compile-flags: --crate-type=rlib -Copt-level=1
//
//@ [amdgpu] compile-flags: --target amdgcn-amd-amdhsa -Ctarget-cpu=gfx900
//@ [amdgpu] needs-llvm-components: amdgpu

//@ [nvptx-pre-llvm-23] compile-flags: --target nvptx64-nvidia-cuda
//@ [nvptx-pre-llvm-23] needs-llvm-components: nvptx
//@ [nvptx-pre-llvm-23] max-llvm-major-version: 22
//@ [nvptx-post-llvm-23] compile-flags: --target nvptx64-nvidia-cuda
//@ [nvptx-post-llvm-23] needs-llvm-components: nvptx
//@ [nvptx-post-llvm-23] min-llvm-version: 23
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

// nvptx-pre-llvm-23: @[[BIG:[^ ]+]] = external addrspace(3) global [0 x i8], align 8
// nvptx-pre-llvm-23: ret { ptr, ptr } { ptr addrspacecast (ptr addrspace(3) @[[BIG]] to ptr), ptr addrspacecast (ptr addrspace(3) @[[BIG]] to ptr) }

// nvptx-post-llvm-23-DAG: @[[SMALL:[^ ]+]] = external addrspace(3) global [0 x i8], align 4
// nvptx-post-llvm-23-DAG: @[[BIG:[^ ]+]] = external addrspace(3) global [0 x i8], align 8
// nvptx-post-llvm-23: ret { ptr, ptr } { ptr addrspacecast (ptr addrspace(3) @[[SMALL]] to ptr), ptr addrspacecast (ptr addrspace(3) @[[BIG]] to ptr) }
#[unsafe(no_mangle)]
pub fn fun() -> (*mut i32, *mut f64) {
    let small = gpu_launch_sized_workgroup_mem::<i32>();
    let big = gpu_launch_sized_workgroup_mem::<f64>(); // Increase alignment to 8
    (small, big)
}
