// Check that pointers are casted to addrspace(0) before they are used

//@ compile-flags: --crate-type=rlib --target=amdgcn-amd-amdhsa -Ctarget-cpu=gfx900 -O
//@ needs-llvm-components: amdgpu
//@ add-minicore
//@ revisions: LLVM21 LLVM22
//@ [LLVM21] max-llvm-major-version: 21
//@ [LLVM22] min-llvm-version: 22
#![feature(no_core)]
#![no_core]

extern crate minicore;

// Make sure that on LLVM 22, the alloca is passed directly to the lifetime intrinsics,
// not the addrspacecast.

// CHECK-LABEL: @ref_of_local
// CHECK: [[alloca:%[0-9]]] = alloca
// CHECK: %i = addrspacecast ptr addrspace(5) [[alloca]] to ptr
// LLVM22: call void @llvm.lifetime.start.p5(ptr addrspace(5) [[alloca]])
// CHECK: call void %f(ptr{{.*}}%i)
// LLVM22: call void @llvm.lifetime.end.p5(ptr addrspace(5) [[alloca]])
#[no_mangle]
pub fn ref_of_local(f: fn(&i32)) {
    let i = 0;
    f(&i);
}

// CHECK-LABEL: @ref_of_global
// CHECK: addrspacecast (ptr addrspace(1) @I to ptr)
#[no_mangle]
pub fn ref_of_global(f: fn(&i32)) {
    #[no_mangle]
    static I: i32 = 0;
    f(&I);
}
