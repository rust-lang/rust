// Tests the amdgpu_dispatch_ptr intrinsic.

//@ compile-flags: --crate-type=rlib --target amdgcn-amd-amdhsa -Ctarget-cpu=gfx900
//@ needs-llvm-components: amdgpu
//@ add-minicore
#![feature(intrinsics, no_core, rustc_attrs)]
#![no_core]

extern crate minicore;

pub struct DispatchPacket {
    pub header: u16,
    pub setup: u16,
    pub workgroup_size_x: u16, // and more
}

#[rustc_intrinsic]
#[rustc_nounwind]
fn amdgpu_dispatch_ptr() -> *const ();

// CHECK: %[[ORIG_PTR:[^ ]+]] = tail call ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
// CHECK: %[[PTR:[^ ]+]] = addrspacecast ptr addrspace(4) %[[ORIG_PTR]] to ptr
// CHECK: ret ptr %[[PTR]]
#[unsafe(no_mangle)]
pub fn get_dispatch_data() -> &'static DispatchPacket {
    unsafe { &*(amdgpu_dispatch_ptr() as *const _) }
}
