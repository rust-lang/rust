//@ compile-flags: -Copt-level=3 -C no-prepopulate-passes
//@ only-64bit (so I don't need to worry about usize)
//@ revisions: aarch64 x86_64
//@ [aarch64] only-aarch64
//@ [aarch64] compile-flags: -C target-feature=+neon
//@ [x86_64] only-x86_64
//@ [x86_64] compile-flags: -C target-feature=+sse2

#![crate_type = "lib"]
#![feature(core_intrinsics)]
#![feature(portable_simd)]

use std::intrinsics::transmute;
use std::simd::{Simd, f32x4, f64x2, i32x4, i64x2};
type PtrX2 = Simd<*const (), 2>;

// These tests use the "C" ABI so that the vectors in question aren't passed and
// returned though memory (as they are in the "Rust" ABI), which greatly
// simplifies seeing the difference between the in-operand cases vs the ones
// that fallback to just using the `LocalKind::Memory` path.

// CHECK-LABEL: <2 x i64> @mixed_int(<4 x i32> %v)
#[no_mangle]
pub extern "C" fn mixed_int(v: i32x4) -> i64x2 {
    // CHECK-NOT: alloca
    // CHECK: %[[RET:.+]] = bitcast <4 x i32> %v to <2 x i64>
    // CHECK: ret <2 x i64> %[[RET]]
    unsafe { transmute(v) }
}

// CHECK-LABEL: <2 x double> @mixed_float(<4 x float> %v)
#[no_mangle]
pub extern "C" fn mixed_float(v: f32x4) -> f64x2 {
    // CHECK-NOT: alloca
    // CHECK: %[[RET:.+]] = bitcast <4 x float> %v to <2 x double>
    // CHECK: ret <2 x double> %[[RET]]
    unsafe { transmute(v) }
}

// CHECK-LABEL: <4 x i32> @float_int_same_lanes(<4 x float> %v)
#[no_mangle]
pub extern "C" fn float_int_same_lanes(v: f32x4) -> i32x4 {
    // CHECK-NOT: alloca
    // CHECK: %[[RET:.+]] = bitcast <4 x float> %v to <4 x i32>
    // CHECK: ret <4 x i32> %[[RET]]
    unsafe { transmute(v) }
}

// CHECK-LABEL: <2 x double> @int_float_same_lanes(<2 x i64> %v)
#[no_mangle]
pub extern "C" fn int_float_same_lanes(v: i64x2) -> f64x2 {
    // CHECK-NOT: alloca
    // CHECK: %[[RET:.+]] = bitcast <2 x i64> %v to <2 x double>
    // CHECK: ret <2 x double> %[[RET]]
    unsafe { transmute(v) }
}

// CHECK-LABEL: <2 x i64> @float_int_widen(<4 x float> %v)
#[no_mangle]
pub extern "C" fn float_int_widen(v: f32x4) -> i64x2 {
    // CHECK-NOT: alloca
    // CHECK: %[[RET:.+]] = bitcast <4 x float> %v to <2 x i64>
    // CHECK: ret <2 x i64> %[[RET]]
    unsafe { transmute(v) }
}

// CHECK-LABEL: <2 x double> @int_float_widen(<4 x i32> %v)
#[no_mangle]
pub extern "C" fn int_float_widen(v: i32x4) -> f64x2 {
    // CHECK-NOT: alloca
    // CHECK: %[[RET:.+]] = bitcast <4 x i32> %v to <2 x double>
    // CHECK: ret <2 x double> %[[RET]]
    unsafe { transmute(v) }
}

// CHECK-LABEL: <4 x i32> @float_int_narrow(<2 x double> %v)
#[no_mangle]
pub extern "C" fn float_int_narrow(v: f64x2) -> i32x4 {
    // CHECK-NOT: alloca
    // CHECK: %[[RET:.+]] = bitcast <2 x double> %v to <4 x i32>
    // CHECK: ret <4 x i32> %[[RET]]
    unsafe { transmute(v) }
}

// CHECK-LABEL: <4 x float> @int_float_narrow(<2 x i64> %v)
#[no_mangle]
pub extern "C" fn int_float_narrow(v: i64x2) -> f32x4 {
    // CHECK-NOT: alloca
    // CHECK: %[[RET:.+]] = bitcast <2 x i64> %v to <4 x float>
    // CHECK: ret <4 x float> %[[RET]]
    unsafe { transmute(v) }
}

// CHECK-LABEL: <2 x ptr> @float_ptr_same_lanes(<2 x double> %v)
#[no_mangle]
pub extern "C" fn float_ptr_same_lanes(v: f64x2) -> PtrX2 {
    // CHECK-NOT: alloca
    // CHECK: %[[TEMP:.+]] = alloca [16 x i8]
    // CHECK-NOT: alloca
    // CHECK: call void @llvm.lifetime.start.p0({{(i64 16, )?}}ptr %[[TEMP]])
    // CHECK: store <2 x double> %v, ptr %[[TEMP]]
    // CHECK: %[[RET:.+]] = load <2 x ptr>, ptr %[[TEMP]]
    // CHECK: call void @llvm.lifetime.end.p0({{(i64 16, )?}}ptr %[[TEMP]])
    // CHECK: ret <2 x ptr> %[[RET]]
    unsafe { transmute(v) }
}

// CHECK-LABEL: <2 x double> @ptr_float_same_lanes(<2 x ptr> %v)
#[no_mangle]
pub extern "C" fn ptr_float_same_lanes(v: PtrX2) -> f64x2 {
    // CHECK-NOT: alloca
    // CHECK: %[[TEMP:.+]] = alloca [16 x i8]
    // CHECK-NOT: alloca
    // CHECK: call void @llvm.lifetime.start.p0({{(i64 16, )?}}ptr %[[TEMP]])
    // CHECK: store <2 x ptr> %v, ptr %[[TEMP]]
    // CHECK: %[[RET:.+]] = load <2 x double>, ptr %[[TEMP]]
    // CHECK: call void @llvm.lifetime.end.p0({{(i64 16, )?}}ptr %[[TEMP]])
    // CHECK: ret <2 x double> %[[RET]]
    unsafe { transmute(v) }
}

// CHECK-LABEL: <2 x ptr> @int_ptr_same_lanes(<2 x i64> %v)
#[no_mangle]
pub extern "C" fn int_ptr_same_lanes(v: i64x2) -> PtrX2 {
    // CHECK-NOT: alloca
    // CHECK: %[[TEMP:.+]] = alloca [16 x i8]
    // CHECK-NOT: alloca
    // CHECK: call void @llvm.lifetime.start.p0({{(i64 16, )?}}ptr %[[TEMP]])
    // CHECK: store <2 x i64> %v, ptr %[[TEMP]]
    // CHECK: %[[RET:.+]] = load <2 x ptr>, ptr %[[TEMP]]
    // CHECK: call void @llvm.lifetime.end.p0({{(i64 16, )?}}ptr %[[TEMP]])
    // CHECK: ret <2 x ptr> %[[RET]]
    unsafe { transmute(v) }
}

// CHECK-LABEL: <2 x i64> @ptr_int_same_lanes(<2 x ptr> %v)
#[no_mangle]
pub extern "C" fn ptr_int_same_lanes(v: PtrX2) -> i64x2 {
    // CHECK-NOT: alloca
    // CHECK: %[[TEMP:.+]] = alloca [16 x i8]
    // CHECK-NOT: alloca
    // CHECK: call void @llvm.lifetime.start.p0({{(i64 16, )?}}ptr %[[TEMP]])
    // CHECK: store <2 x ptr> %v, ptr %[[TEMP]]
    // CHECK: %[[RET:.+]] = load <2 x i64>, ptr %[[TEMP]]
    // CHECK: call void @llvm.lifetime.end.p0({{(i64 16, )?}}ptr %[[TEMP]])
    // CHECK: ret <2 x i64> %[[RET]]
    unsafe { transmute(v) }
}

// CHECK-LABEL: <2 x ptr> @float_ptr_widen(<4 x float> %v)
#[no_mangle]
pub extern "C" fn float_ptr_widen(v: f32x4) -> PtrX2 {
    // CHECK-NOT: alloca
    // CHECK: %[[TEMP:.+]] = alloca [16 x i8]
    // CHECK-NOT: alloca
    // CHECK: call void @llvm.lifetime.start.p0({{(i64 16, )?}}ptr %[[TEMP]])
    // CHECK: store <4 x float> %v, ptr %[[TEMP]]
    // CHECK: %[[RET:.+]] = load <2 x ptr>, ptr %[[TEMP]]
    // CHECK: call void @llvm.lifetime.end.p0({{(i64 16, )?}}ptr %[[TEMP]])
    // CHECK: ret <2 x ptr> %[[RET]]
    unsafe { transmute(v) }
}

// CHECK-LABEL: <2 x ptr> @int_ptr_widen(<4 x i32> %v)
#[no_mangle]
pub extern "C" fn int_ptr_widen(v: i32x4) -> PtrX2 {
    // CHECK-NOT: alloca
    // CHECK: %[[TEMP:.+]] = alloca [16 x i8]
    // CHECK-NOT: alloca
    // CHECK: call void @llvm.lifetime.start.p0({{(i64 16, )?}}ptr %[[TEMP]])
    // CHECK: store <4 x i32> %v, ptr %[[TEMP]]
    // CHECK: %[[RET:.+]] = load <2 x ptr>, ptr %[[TEMP]]
    // CHECK: call void @llvm.lifetime.end.p0({{(i64 16, )?}}ptr %[[TEMP]])
    // CHECK: ret <2 x ptr> %[[RET]]
    unsafe { transmute(v) }
}
