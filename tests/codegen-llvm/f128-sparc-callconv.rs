//! Verify that Rust implements the expected calling convention for `f128`

//@ add-minicore
//@ revisions: sparc-none sparc-linux
//@ [sparc-none] compile-flags: --target sparc-unknown-none-elf
//@ [sparc-linux] compile-flags: --target sparc-unknown-linux-gnu
//@ compile-flags: -Copt-level=3
//@ needs-llvm-components: sparc

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
    // an f128 is passed via the stack
    // CHECK-SAME: ptr {{.*}}byval([16 x i8]
    // CHECK: call void @extern_call
    unsafe { extern_call(arg1) };
}

// Check that we produce the correct return ABI
#[no_mangle]
pub extern "C" fn ret(_arg0: u32, arg1: f128) -> f128 {
    // CHECK-LABEL: @ret(
    // and an f128 is returned via the stack
    // CHECK-SAME: sret([16 x i8])
    // CHECK: %0 = load fp128, ptr %arg1
    // CHECK-NEXT: store fp128 %0, ptr %_0
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
