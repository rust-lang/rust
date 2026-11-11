//@ add-minicore
//@ revisions: amdgpu nvptx
//@[nvptx] compile-flags: -Copt-level=0 -Zunstable-options -Zoffload=Device --target nvptx64-nvidia-cuda --crate-type=rlib
//@[nvptx] needs-llvm-components: nvptx
//@[amdgpu] compile-flags: -Copt-level=0 -Zunstable-options -Zoffload=Device --target amdgcn-amd-amdhsa -Ctarget-cpu=gfx900 --crate-type=rlib
//@[amdgpu] needs-llvm-components: amdgpu
//@ no-prefer-dynamic
//@ needs-offload

// This test verifies that the offload intrinsic is properly handling scalar args on the device,
// replacing the args by i64 and then trunc and cast them to the original type

#![feature(abi_gpu_kernel, rustc_attrs, no_core)]
#![no_core]

extern crate minicore;

// CHECK: ; Function Attrs
// nvptx-NEXT: define ptx_kernel void @foo(ptr %dyn_ptr, ptr %0, i64 %1)
// amdgpu-NEXT: define amdgpu_kernel void @foo(ptr %dyn_ptr, ptr %0, i64 %1)
// CHECK-NEXT: entry:
// CHECK-NEXT: %2 = trunc i64 %1 to i32
// CHECK-NEXT: %3 = bitcast i32 %2 to float
// CHECK-NEXT: br label %start
// CHECK: start:
// CHECK-NEXT: store float %3, ptr %0, align 4
// CHECK-NEXT: ret void
// CHECK-NEXT: }

#[unsafe(no_mangle)]
#[rustc_offload_kernel]
pub unsafe extern "gpu-kernel" fn foo(x: *mut f32, k: f32) {
    unsafe {
        *x = k;
    };
}
