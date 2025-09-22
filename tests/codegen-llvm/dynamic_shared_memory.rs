// Checks that dynamic_shared_memory works.

//@ revisions: amdgpu nvptx x86
//@ compile-flags: --crate-type=rlib
//
//@ [amdgpu] compile-flags: --target amdgcn-amd-amdhsa -Ctarget-cpu=gfx900
//@ [amdgpu] only-amdgpu
//@ [amdgpu] needs-llvm-components: amdgpu
//@ [nvptx] compile-flags: --target nvptx64-nvidia-cuda
//@ [nvptx] only-nvptx64
//@ [nvptx] needs-llvm-components: nvptx
//@ [x86] compile-flags: --target x86_64-unknown-linux-gnu
//@ [x86] only-x86_64
//@ [x86] needs-llvm-components: x86
//@ [x86] should-fail
#![feature(core_intrinsics, dynamic_shared_memory)]
#![no_std]

use core::intrinsics::dynamic_shared_memory;

// CHECK: @dynamic_shared_memory = external addrspace(3) global [0 x i8], align 8
// CHECK: ret ptr addrspacecast (ptr addrspace(3) @dynamic_shared_memory to ptr)
pub fn fun() -> *mut i32 {
    let res = dynamic_shared_memory::<i32>();
    dynamic_shared_memory::<f64>(); // Increase alignment to 8
    res
}
