//@ revisions: OPT0 OPT3
//@ [OPT0] compile-flags: -Copt-level=0
//@ [OPT3] compile-flags: -Copt-level=3
//@ compile-flags: -C no-prepopulate-passes
//@ only-64bit (so I don't need to worry about usize)
// ignore-tidy-linelength (the memcpy calls get long)

#![crate_type = "lib"]
#![feature(core_intrinsics)]

use std::intrinsics::typed_swap_nonoverlapping;

// CHECK-LABEL: @swap_unit(
#[no_mangle]
pub unsafe fn swap_unit(x: &mut (), y: &mut ()) {
    // CHECK: start
    // CHECK-NEXT: ret void
    typed_swap_nonoverlapping(x, y)
}

// CHECK-LABEL: @swap_i32(
#[no_mangle]
pub unsafe fn swap_i32(x: &mut i32, y: &mut i32) {
    // CHECK-NOT: alloca

    // CHECK: %[[TEMP:.+]] = load i32, ptr %x, align 4
    // OPT3-SAME: !noundef
    // OPT0: %[[TEMP2:.+]] = load i32, ptr %y, align 4
    // OPT0: store i32 %[[TEMP2]], ptr %x, align 4
    // OPT0-NOT: memcpy
    // OPT3-NOT: load
    // OPT3: call void @llvm.memcpy.p0.p0.i64(ptr align 4 %x, ptr align 4 %y, i64 4, i1 false)
    // CHECK: store i32 %[[TEMP]], ptr %y, align 4
    // CHECK: ret void
    typed_swap_nonoverlapping(x, y)
}

// CHECK-LABEL: @swap_pair(
#[no_mangle]
pub unsafe fn swap_pair(x: &mut (i32, u32), y: &mut (i32, u32)) {
    // CHECK-NOT: alloca

    // CHECK: load i32
    // OPT3-SAME: !noundef
    // CHECK: load i32
    // OPT3-SAME: !noundef
    // CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 4 %x, ptr align 4 %y, i64 8, i1 false)
    // CHECK: store i32
    // CHECK: store i32
    typed_swap_nonoverlapping(x, y)
}

// CHECK-LABEL: @swap_str(
#[no_mangle]
pub unsafe fn swap_str<'a>(x: &mut &'a str, y: &mut &'a str) {
    // CHECK-NOT: alloca

    // CHECK: load ptr
    // OPT3-SAME: !nonnull
    // OPT3-SAME: !noundef
    // CHECK: load i64
    // OPT3-SAME: !noundef
    // CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 8 %x, ptr align 8 %y, i64 16, i1 false)
    // CHECK: store ptr
    // CHECK: store i64
    typed_swap_nonoverlapping(x, y)
}

// OPT0-LABEL: @swap_string(
#[no_mangle]
pub unsafe fn swap_string(x: &mut String, y: &mut String) {
    // OPT0: %[[TEMP:.+]] = alloca {{.+}}, align 8
    // OPT0: call void @llvm.memcpy.p0.p0.i64(ptr align 8 %[[TEMP]], ptr align 8 %x, i64 24, i1 false)
    // OPT0: call void @llvm.memcpy.p0.p0.i64(ptr align 8 %x, ptr align 8 %y, i64 24, i1 false)
    // OPT0: call void @llvm.memcpy.p0.p0.i64(ptr align 8 %y, ptr align 8 %[[TEMP]], i64 24, i1 false)
    typed_swap_nonoverlapping(x, y)
}
