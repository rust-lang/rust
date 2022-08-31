// ignore-test

// WHY IS THIS TEST BEING IGNORED:
//
// This test depends on characteristics of how the stdlib was compiled,
// namely that sufficient inlining occurred to ensure that the call to
// `std::mem::replace` boils down to just two calls of `llvm.memcpy`.
//
// But the MIR inlining policy is in flux as of 1.64-beta, and the intermittent
// breakage of this test that results is causing problems for people trying to
// do development.

// This test ensures that `mem::replace::<T>` only ever calls `@llvm.memcpy`
// with `size_of::<T>()` as the size, and never goes through any wrapper that
// may e.g. multiply `size_of::<T>()` with a variable "count" (which is only
// known to be `1` after inlining).

// compile-flags: -C no-prepopulate-passes -Zinline-mir=no

#![crate_type = "lib"]

pub fn replace_byte(dst: &mut u8, src: u8) -> u8 {
    std::mem::replace(dst, src)
}

// NOTE(eddyb) the `CHECK-NOT`s ensure that the only calls of `@llvm.memcpy` in
// the entire output, are the two direct calls we want, from `ptr::replace`.

// CHECK-NOT: call void @llvm.memcpy
// CHECK: ; core::mem::replace
// CHECK-NOT: call void @llvm.memcpy
// CHECK: call void @llvm.memcpy.{{.+}}({{i8\*|ptr}} align 1 %{{.*}}, {{i8\*|ptr}} align 1 %dest, i{{.*}} 1, i1 false)
// CHECK-NOT: call void @llvm.memcpy
// CHECK: call void @llvm.memcpy.{{.+}}({{i8\*|ptr}} align 1 %dest, {{i8\*|ptr}} align 1 %src{{.*}}, i{{.*}} 1, i1 false)
// CHECK-NOT: call void @llvm.memcpy
