// This test ensures that `mem::replace::<T>` only ever calls `@llvm.memcpy`
// with `size_of::<T>()` as the size, and never goes through any wrapper that
// may e.g. multiply `size_of::<T>()` with a variable "count" (which is only
// known to be `1` after inlining).

// compile-flags: -C no-prepopulate-passes -Zinline-mir=no
// ignore-debug: the debug assertions get in the way

#![crate_type = "lib"]

pub fn replace_byte(dst: &mut u8, src: u8) -> u8 {
    std::mem::replace(dst, src)
}

// NOTE(eddyb) the `CHECK-NOT`s ensure that the only calls of `@llvm.memcpy` in
// the entire output, are the direct calls we want, from `ptr::replace`.

// CHECK-NOT: call void @llvm.memcpy

// For a small type, we expect one each of `load`/`store`/`memcpy` instead
// CHECK-LABEL: define internal noundef i8 @{{.+}}mem{{.+}}replace
    // CHECK-NOT: alloca
    // CHECK: alloca i8
    // CHECK-NOT: alloca
    // CHECK-NOT: call void @llvm.memcpy
    // CHECK: load i8
    // CHECK-NOT: call void @llvm.memcpy
    // CHECK: store i8
    // CHECK-NOT: call void @llvm.memcpy
    // CHECK: call void @llvm.memcpy.{{.+}}({{i8\*|ptr}} align 1 %{{.*}}, {{i8\*|ptr}} align 1 %{{.*}}, i{{.*}} 1, i1 false)
    // CHECK-NOT: call void @llvm.memcpy

// CHECK-NOT: call void @llvm.memcpy
