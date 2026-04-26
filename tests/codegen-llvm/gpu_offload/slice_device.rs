//@ add-minicore
//@ revisions: amdgpu nvptx
//@[nvptx] compile-flags: -Copt-level=3 -Zunstable-options -Zoffload=Device --target nvptx64-nvidia-cuda --crate-type=rlib
//@[nvptx] needs-llvm-components: nvptx
//@[amdgpu] compile-flags: -Copt-level=3 -Zunstable-options -Zoffload=Device --target amdgcn-amd-amdhsa -Ctarget-cpu=gfx900 --crate-type=rlib
//@[amdgpu] needs-llvm-components: amdgpu
//@ no-prefer-dynamic
//@ needs-offload

#![feature(abi_gpu_kernel, rustc_attrs, no_core)]
#![no_core]

extern crate minicore;

// CHECK: ; Function Attrs
// nvptx-NEXT: define ptx_kernel void @foo
// amdgpu-NEXT: define amdgpu_kernel void @foo
// CHECK-SAME: ptr readnone captures(none) %dyn_ptr
// nvptx-SAME: [2 x i64] %0
// amdgpu-SAME: ptr noalias {{.*}} %0, i64 {{.*}} %1
// CHECK-NEXT: entry:
// CHECK-NEXT: ret void
// CHECK-NEXT: }

#[unsafe(no_mangle)]
#[rustc_offload_kernel]
pub unsafe extern "gpu-kernel" fn foo(x: &[f32]) {}
