// Test that temporary allocas used for call arguments have their lifetimes described by
// intrinsics.
//
//@ add-core-stubs
//@ compile-flags: -Copt-level=1 -Cno-prepopulate-passes --crate-type=lib --target i686-unknown-linux-gnu
//@ needs-llvm-components: x86
#![feature(no_core)]
#![no_std]
#![no_core]
extern crate minicore;
use minicore::*;

// Const operand. Regression test for #98156.
//
// CHECK-LABEL: define void @const_indirect(
// CHECK-NEXT: start:
// CHECK-NEXT: [[B:%.*]] = alloca
// CHECK-NEXT: [[A:%.*]] = alloca
// CHECK-NEXT: call void @llvm.lifetime.start.p0(i64 4096, ptr [[A]])
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[A]], ptr align 4 {{.*}}, i32 4096, i1 false)
// CHECK-NEXT: call void %h(ptr {{.*}} [[A]])
// CHECK-NEXT: call void @llvm.lifetime.end.p0(i64 4096, ptr [[A]])
// CHECK-NEXT: call void @llvm.lifetime.start.p0(i64 4096, ptr [[B]])
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[B]], ptr align 4 {{.*}}, i32 4096, i1 false)
// CHECK-NEXT: call void %h(ptr {{.*}} [[B]])
// CHECK-NEXT: call void @llvm.lifetime.end.p0(i64 4096, ptr [[B]])
#[no_mangle]
pub fn const_indirect(h: extern "C" fn([u32; 1024])) {
    const C: [u32; 1024] = [0; 1024];
    h(C);
    h(C);
}

#[repr(C)]
pub struct Str {
    pub ptr: *const u8,
    pub len: usize,
}

// Pair of immediates. Regression test for #132014.
//
// CHECK-LABEL: define void @immediate_indirect(ptr {{.*}}%s.0, i32 {{.*}}%s.1, ptr {{.*}}%g)
// CHECK-NEXT: start:
// CHECK-NEXT: [[A:%.*]] = alloca
// CHECK-NEXT: call void @llvm.lifetime.start.p0(i64 8, ptr [[A]])
// CHECK-NEXT: store ptr %s.0, ptr [[A]]
// CHECK-NEXT: [[B:%.]] = getelementptr inbounds i8, ptr [[A]], i32 4
// CHECK-NEXT: store i32 %s.1, ptr [[B]]
// CHECK-NEXT: call void %g(ptr {{.*}} [[A]])
// CHECK-NEXT: call void @llvm.lifetime.end.p0(i64 8, ptr [[A]])
#[no_mangle]
pub fn immediate_indirect(s: Str, g: extern "C" fn(Str)) {
    g(s);
}

// Indirect argument with a higher alignment requirement than the type's.
//
// CHECK-LABEL: define void @align_indirect(ptr{{.*}} align 1{{.*}} %a, ptr{{.*}} %fun)
// CHECK-NEXT: start:
// CHECK-NEXT: [[A:%.*]] = alloca [1024 x i8], align 4
// CHECK-NEXT: call void @llvm.lifetime.start.p0(i64 1024, ptr [[A]])
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[A]], ptr align 1 %a, i32 1024, i1 false)
// CHECK-NEXT: call void %fun(ptr {{.*}} [[A]])
// CHECK-NEXT: call void @llvm.lifetime.end.p0(i64 1024, ptr [[A]])
#[no_mangle]
pub fn align_indirect(a: [u8; 1024], fun: extern "C" fn([u8; 1024])) {
    fun(a);
}
