//! Verify that Rust implements the expected calling convention for `f128`

//@ add-core-stubs
//@ compile-flags: -Copt-level=3 --target wasm32-wasip1
//@ needs-llvm-components: webassembly

#![crate_type = "lib"]
#![no_std]
#![no_core]
#![feature(no_core, lang_items, f128)]

extern crate minicore;

extern "C" {
    fn extern_call(arg0: f128);
    fn extern_ret() -> f128;
}

#[no_mangle]
pub extern "C" fn pass(_arg0: u32, arg1: f128) {
    // CHECK-LABEL: @pass(
    // an f128 is passed via registers
    // CHECK-SAME: fp128 noundef %arg1
    // CHECK: call void @extern_call
    unsafe { extern_call(arg1) };
}

// Check that we produce the correct return ABI
#[no_mangle]
pub extern "C" fn ret(_arg0: u32, arg1: f128) -> f128 {
    // CHECK-LABEL: @ret(
    // but an f128 is returned via the stack
    // CHECK-SAME: sret
    // CHECK: store fp128 %arg1
    // CHECK-NEXT: ret void
    arg1
}

// Check that we consume the correct return ABI
#[no_mangle]
pub extern "C" fn forward(dst: *mut f128) {
    // CHECK-LABEL: @forward
    // CHECK-SAME: ptr{{.*}} %dst)
    // without optimizatons, an intermediate alloca is used
    // CHECK: call void @extern_ret
    // CHECK: store fp128
    // CHECK: ret void
    unsafe { *dst = extern_ret() };
}
