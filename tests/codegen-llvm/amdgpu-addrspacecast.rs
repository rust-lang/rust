// Check that pointers are casted to addrspace(0) before they are used

//@ compile-flags: --crate-type=rlib --target=amdgcn-amd-amdhsa -Ctarget-cpu=gfx900
//@ needs-llvm-components: amdgpu
//@ add-core-stubs
#![feature(no_core)]
#![no_core]

extern crate minicore;

// CHECK-LABEL: @ref_of_local
// CHECK: [[alloca:%[0-9]]] = alloca
// CHECK: %i = addrspacecast ptr addrspace(5) [[alloca]] to ptr
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
