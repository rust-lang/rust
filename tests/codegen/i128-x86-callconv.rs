//! Verify that Rust implements the expected calling convention for `i128`/`u128`.

// Eliminate intermediate instructions during `nop` tests
//@ compile-flags: -Copt-level=1

//@ add-core-stubs
//@ revisions: MSVC MINGW
//@ [MSVC] needs-llvm-components: x86
//@ [MINGW] needs-llvm-components: x86
//@ [MSVC] compile-flags: --target x86_64-pc-windows-msvc
//@ [MINGW] compile-flags: --target x86_64-pc-windows-gnu
//@ [MSVC] filecheck-flags: --check-prefix=WIN
//@ [MINGW] filecheck-flags: --check-prefix=WIN

#![crate_type = "lib"]
#![no_std]
#![no_core]
#![feature(no_core, lang_items)]

extern crate minicore;

extern "C" {
    fn extern_call(arg0: i128);
    fn extern_ret() -> i128;
}

#[no_mangle]
pub extern "C" fn pass(_arg0: u32, arg1: i128) {
    // CHECK-LABEL: @pass(
    // i128 is passed indirectly on Windows. It should load the pointer to the stack and pass
    // a pointer to that allocation.
    // WIN-SAME: %_arg0, ptr{{.*}} %arg1)
    // WIN: [[PASS:%[_0-9]+]] = alloca [16 x i8], align 16
    // WIN: [[LOADED:%[_0-9]+]] = load i128, ptr %arg1
    // WIN: store i128 [[LOADED]], ptr [[PASS]]
    // WIN: call void @extern_call
    unsafe { extern_call(arg1) };
}

// Check that we produce the correct return ABI
#[no_mangle]
pub extern "C" fn ret(_arg0: u32, arg1: i128) -> i128 {
    // CHECK-LABEL: @ret(
    // i128 is returned in xmm0 on Windows
    // FIXME(#134288): This may change for the `-msvc` targets in the future.
    // WIN-SAME: i32{{.*}} %_arg0, ptr{{.*}} %arg1)
    // WIN: [[LOADED:%[_0-9]+]] = load <16 x i8>, ptr %arg1
    // WIN-NEXT: ret <16 x i8> [[LOADED]]
    arg1
}

// Check that we consume the correct return ABI
#[no_mangle]
pub extern "C" fn forward(dst: *mut i128) {
    // CHECK-LABEL: @forward
    // WIN-SAME: ptr{{.*}} %dst)
    // WIN: [[RETURNED:%[_0-9]+]] = tail call <16 x i8> @extern_ret()
    // WIN: store <16 x i8> [[RETURNED]], ptr %dst
    // WIN: ret void
    unsafe { *dst = extern_ret() };
}

#[repr(C)]
struct RetAggregate {
    a: i32,
    b: i128,
}

#[no_mangle]
pub extern "C" fn ret_aggregate(_arg0: u32, arg1: i128) -> RetAggregate {
    // CHECK-LABEL: @ret_aggregate(
    // Aggregates should also be returned indirectly
    // WIN-SAME: ptr{{.*}}sret([32 x i8]){{.*}}[[RET:%[_0-9]+]], i32{{.*}}%_arg0, ptr{{.*}}%arg1)
    // WIN: [[LOADED:%[_0-9]+]] = load i128, ptr %arg1
    // WIN: [[GEP:%[_0-9]+]] = getelementptr{{.*}}, ptr [[RET]]
    // WIN: store i128 [[LOADED]], ptr [[GEP]]
    // WIN: ret void
    RetAggregate { a: 1, b: arg1 }
}
