// This test ensures that `mem::replace::<T>` only ever calls `@llvm.memcpy`
// with `size_of::<T>()` as the size, and never goes through any wrapper that
// may e.g. multiply `size_of::<T>()` with a variable "count" (which is only
// known to be `1` after inlining).

//@ compile-flags: -C no-prepopulate-passes -Zinline-mir=no
//@ ignore-std-debug-assertions
// Reason: precondition checks in ptr::read make them a bad candidate for MIR inlining
//@ needs-deterministic-layouts

#![crate_type = "lib"]

#[repr(C, align(8))]
pub struct Big([u64; 7]);
pub fn replace_big(dst: &mut Big, src: Big) -> Big {
    // Back in 1.68, this emitted six `memcpy`s.
    // `read_via_copy` in 1.69 got that down to three.
    // `write_via_move` and nvro get this down to the essential two.
    std::mem::replace(dst, src)
}

// NOTE(eddyb) the `CHECK-NOT`s ensure that the only calls of `@llvm.memcpy` in
// the entire output, are the direct calls we want, from `ptr::replace`.

// CHECK-NOT: call void @llvm.memcpy

// For a large type, we expect exactly three `memcpy`s
// CHECK-LABEL: define internal void @{{.+}}mem{{.+}}replace{{.+}}sret([56 x i8])
// CHECK-NOT: call void @llvm.memcpy
// CHECK: call void @llvm.memcpy.{{.+}}(ptr align 8 %result, ptr align 8 %dest, i{{.*}} 56, i1 false)
// CHECK-NOT: call void @llvm.memcpy
// CHECK: call void @llvm.memcpy.{{.+}}(ptr align 8 %dest, ptr align 8 %src, i{{.*}} 56, i1 false)
// CHECK-NOT: call void @llvm.memcpy

// CHECK-NOT: call void @llvm.memcpy
