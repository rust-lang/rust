// Checks that when compiling for GPU targets, the convergent attribute
// is added to function declarations and definitions.

//@ add-minicore
//@ revisions: amdgpu nvptx
//@ [amdgpu] compile-flags: --crate-type=rlib --target=amdgcn-amd-amdhsa -Ctarget-cpu=gfx900
//@ [amdgpu] needs-llvm-components: amdgpu
//@ [nvptx] compile-flags: --crate-type=rlib --target=nvptx64-nvidia-cuda
//@ [nvptx] needs-llvm-components: nvptx
#![feature(no_core, lang_items, abi_gpu_kernel)]
#![no_core]

extern crate minicore;
use minicore::*;

extern "C" {
    fn ext();
}

// CHECK: define {{.*}}_kernel void @fun(i32{{.*}}) unnamed_addr #[[ATTR:[0-9]+]] {
// CHECK: declare void @ext() unnamed_addr #[[ATTR]]
// CHECK: attributes #[[ATTR]] = {{.*}} convergent
#[no_mangle]
pub extern "gpu-kernel" fn fun(_: i32) {
    unsafe {
        ext();
    }
}
