// This test ensures that `mem::replace::<T>` only ever calls `@llvm.memcpy`
// with `size_of::<T>()` as the size, and never goes through any wrapper that
// may e.g. multiply `size_of::<T>()` with a variable "count" (which is only
// known to be `1` after inlining).

// compile-flags: -C no-prepopulate-passes -Zinline-mir=no
// ignore-debug: the debug assertions get in the way

#![crate_type = "lib"]

#[repr(C, align(8))]
pub struct Big([u64; 7]);
pub fn replace_big(dst: &mut Big, src: Big) -> Big {
    // Before the `read_via_copy` intrinsic, this emitted six `memcpy`s.
    std::mem::replace(dst, src)
}

// NOTE(eddyb) the `CHECK-NOT`s ensure that the only calls of `@llvm.memcpy` in
// the entire output, are the direct calls we want, from `ptr::replace`.

// CHECK-NOT: call void @llvm.memcpy

// For a large type, we expect exactly three `memcpy`s
// CHECK-LABEL: define internal void @{{.+}}mem{{.+}}replace{{.+}}sret(%Big)
    // CHECK-NOT: alloca
    // CHECK: alloca %Big
    // CHECK-NOT: alloca
    // CHECK-NOT: call void @llvm.memcpy
    // CHECK: call void @llvm.memcpy.{{.+}}({{i8\*|ptr}} align 8 %{{.*}}, {{i8\*|ptr}} align 8 %{{.*}}, i{{.*}} 56, i1 false)
    // CHECK-NOT: call void @llvm.memcpy
    // CHECK: call void @llvm.memcpy.{{.+}}({{i8\*|ptr}} align 8 %{{.*}}, {{i8\*|ptr}} align 8 %{{.*}}, i{{.*}} 56, i1 false)
    // CHECK-NOT: call void @llvm.memcpy
    // CHECK: call void @llvm.memcpy.{{.+}}({{i8\*|ptr}} align 8 %{{.*}}, {{i8\*|ptr}} align 8 %{{.*}}, i{{.*}} 56, i1 false)
    // CHECK-NOT: call void @llvm.memcpy

// CHECK-NOT: call void @llvm.memcpy
