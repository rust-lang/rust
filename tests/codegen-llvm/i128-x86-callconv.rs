//! Verify that Rust implements the expected calling convention for `i128`/`u128`.

// Eliminate intermediate instructions during `nop` tests
//@ compile-flags: -Copt-level=1

//@ add-core-stubs
//@ revisions: MSVC MINGW softfloat
//@ [MSVC] needs-llvm-components: x86
//@ [MSVC] compile-flags: --target x86_64-pc-windows-msvc
// Use `WIN` as a common prefix for MSVC and MINGW but *not* the softfloat test.
//@ [MSVC] filecheck-flags: --check-prefix=WIN
//@ [MINGW] needs-llvm-components: x86
//@ [MINGW] compile-flags: --target x86_64-pc-windows-gnu
//@ [MINGW] filecheck-flags: --check-prefix=WIN
// The `x86_64-unknown-uefi` target also uses the Windows calling convention,
// but does not have SSE registers available.
//@ [softfloat] needs-llvm-components: x86
//@ [softfloat] compile-flags: --target x86_64-unknown-uefi

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
    // a pointer to that allocation. The softfloat ABI works the same.
    // CHECK-SAME: %_arg0, ptr{{.*}} %arg1)
    // CHECK: [[PASS:%[_0-9]+]] = alloca [16 x i8], align 16
    // CHECK: [[LOADED:%[_0-9]+]] = load i128, ptr %arg1
    // CHECK: store i128 [[LOADED]], ptr [[PASS]]
    // CHECK: call void @extern_call
    unsafe { extern_call(arg1) };
}

// Check that we produce the correct return ABI
#[no_mangle]
pub extern "C" fn ret(_arg0: u32, arg1: i128) -> i128 {
    // WIN-LABEL: @ret(
    // i128 is returned in xmm0 on Windows
    // FIXME(#134288): This may change for the `-msvc` targets in the future.
    // WIN-SAME: i32{{.*}} %_arg0, ptr{{.*}} %arg1)
    // WIN: [[LOADED:%[_0-9]+]] = load <16 x i8>, ptr %arg1
    // WIN-NEXT: ret <16 x i8> [[LOADED]]
    // The softfloat ABI returns this indirectly.
    // softfloat-LABEL: i128 @ret(i32{{.*}} %_arg0, ptr{{.*}} %arg1)
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
    // softfloat: [[RETURNED:%[_0-9]+]] = tail call {{.*}}i128 @extern_ret()
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
    // CHECK-SAME: ptr{{.*}}sret([32 x i8]){{.*}}[[RET:%[_0-9]+]], i32{{.*}}%_arg0, ptr{{.*}}%arg1)
    // CHECK: [[LOADED:%[_0-9]+]] = load i128, ptr %arg1
    // CHECK: [[GEP:%[_0-9]+]] = getelementptr{{.*}}, ptr [[RET]]
    // CHECK: store i128 [[LOADED]], ptr [[GEP]]
    // CHECK: ret void
    RetAggregate { a: 1, b: arg1 }
}
