//@ add-core-stubs
//@ compile-flags: -Copt-level=0 -Cdebuginfo=0 --target riscv64gc-unknown-linux-gnu
//@ needs-llvm-components: riscv

#![feature(no_core, lang_items)]
#![no_std]
#![no_core]
#![crate_type = "lib"]

extern crate minicore;
use minicore::*;

#[repr(C, align(64))]
struct Aligned(f64);

#[repr(C, align(64))]
struct AlignedPair(f32, f64);

impl Copy for Aligned {}
impl Copy for AlignedPair {}

// CHECK-LABEL: define double @read_aligned
#[unsafe(no_mangle)]
pub extern "C" fn read_aligned(x: &Aligned) -> Aligned {
    // CHECK: %[[TEMP:.*]] = alloca [64 x i8], align 64
    // CHECK-NEXT: call void @llvm.memcpy.p0.p0.i64(ptr align 64 %[[TEMP]], ptr align 64 %[[PTR:.*]], i64 64, i1 false)
    // CHECK-NEXT: %[[RES:.*]] = load double, ptr %[[TEMP]], align 64
    // CHECK-NEXT: ret double %[[RES]]
    *x
}

// CHECK-LABEL: define { float, double } @read_aligned_pair
#[unsafe(no_mangle)]
pub extern "C" fn read_aligned_pair(x: &AlignedPair) -> AlignedPair {
    // CHECK: %[[TEMP:.*]] = alloca [64 x i8], align 64
    // CHECK-NEXT: call void @llvm.memcpy.p0.p0.i64(ptr align 64 %[[TEMP]], ptr align 64 %[[PTR:.*]], i64 64, i1 false)
    // CHECK-NEXT: %[[FIRST:.*]] = load float, ptr %[[TEMP]], align 64
    // CHECK-NEXT: %[[SECOND_PTR:.*]] = getelementptr inbounds i8, ptr %[[TEMP]], i64 8
    // CHECK-NEXT: %[[SECOND:.*]] = load double, ptr %[[SECOND_PTR]], align 8
    // CHECK-NEXT: %[[RES1:.*]] = insertvalue { float, double } poison, float %[[FIRST]], 0
    // CHECK-NEXT: %[[RES2:.*]] = insertvalue { float, double } %[[RES1]], double %[[SECOND]], 1
    // CHECK-NEXT: ret { float, double } %[[RES2]]
    *x
}
