//@ compile-flags: -Copt-level=3 -C no-prepopulate-passes
//@ only-64bit (so I don't need to worry about usize)
//@ revisions: aarch64 x86_64
//@ [aarch64] only-aarch64
//@ [aarch64] compile-flags: -C target-feature=+neon
//@ [x86_64] only-x86_64
//@ [x86_64] compile-flags: -C target-feature=+sse2

#![crate_type = "lib"]
#![feature(core_intrinsics)]
#![feature(repr_simd)]

use std::intrinsics::transmute;

// These tests use the "C" ABI so that the vectors in question aren't passed and
// returned though memory (as they are in the "Rust" ABI), which greatly
// simplifies seeing the difference between the in-operand cases vs the ones
// that fallback to just using the `LocalKind::Memory` path.

#[repr(simd)]
pub struct I32X4([i32; 4]);
#[repr(simd)]
pub struct I64X2([i64; 2]);
#[repr(simd)]
pub struct F32X4([f32; 4]);
#[repr(simd)]
pub struct F64X2([f64; 2]);
#[repr(simd)]
pub struct PtrX2([*const (); 2]);

// CHECK-LABEL: <2 x i64> @mixed_int(<4 x i32> %v)
#[no_mangle]
pub extern "C" fn mixed_int(v: I32X4) -> I64X2 {
    // CHECK-NOT: alloca
    // CHECK: %[[RET:.+]] = bitcast <4 x i32> %v to <2 x i64>
    // CHECK: ret <2 x i64> %[[RET]]
    unsafe { transmute(v) }
}

// CHECK-LABEL: <2 x double> @mixed_float(<4 x float> %v)
#[no_mangle]
pub extern "C" fn mixed_float(v: F32X4) -> F64X2 {
    // CHECK-NOT: alloca
    // CHECK: %[[RET:.+]] = bitcast <4 x float> %v to <2 x double>
    // CHECK: ret <2 x double> %[[RET]]
    unsafe { transmute(v) }
}

// CHECK-LABEL: <4 x i32> @float_int_same_lanes(<4 x float> %v)
#[no_mangle]
pub extern "C" fn float_int_same_lanes(v: F32X4) -> I32X4 {
    // CHECK-NOT: alloca
    // CHECK: %[[RET:.+]] = bitcast <4 x float> %v to <4 x i32>
    // CHECK: ret <4 x i32> %[[RET]]
    unsafe { transmute(v) }
}

// CHECK-LABEL: <2 x double> @int_float_same_lanes(<2 x i64> %v)
#[no_mangle]
pub extern "C" fn int_float_same_lanes(v: I64X2) -> F64X2 {
    // CHECK-NOT: alloca
    // CHECK: %[[RET:.+]] = bitcast <2 x i64> %v to <2 x double>
    // CHECK: ret <2 x double> %[[RET]]
    unsafe { transmute(v) }
}

// CHECK-LABEL: <2 x i64> @float_int_widen(<4 x float> %v)
#[no_mangle]
pub extern "C" fn float_int_widen(v: F32X4) -> I64X2 {
    // CHECK-NOT: alloca
    // CHECK: %[[RET:.+]] = bitcast <4 x float> %v to <2 x i64>
    // CHECK: ret <2 x i64> %[[RET]]
    unsafe { transmute(v) }
}

// CHECK-LABEL: <2 x double> @int_float_widen(<4 x i32> %v)
#[no_mangle]
pub extern "C" fn int_float_widen(v: I32X4) -> F64X2 {
    // CHECK-NOT: alloca
    // CHECK: %[[RET:.+]] = bitcast <4 x i32> %v to <2 x double>
    // CHECK: ret <2 x double> %[[RET]]
    unsafe { transmute(v) }
}

// CHECK-LABEL: <4 x i32> @float_int_narrow(<2 x double> %v)
#[no_mangle]
pub extern "C" fn float_int_narrow(v: F64X2) -> I32X4 {
    // CHECK-NOT: alloca
    // CHECK: %[[RET:.+]] = bitcast <2 x double> %v to <4 x i32>
    // CHECK: ret <4 x i32> %[[RET]]
    unsafe { transmute(v) }
}

// CHECK-LABEL: <4 x float> @int_float_narrow(<2 x i64> %v)
#[no_mangle]
pub extern "C" fn int_float_narrow(v: I64X2) -> F32X4 {
    // CHECK-NOT: alloca
    // CHECK: %[[RET:.+]] = bitcast <2 x i64> %v to <4 x float>
    // CHECK: ret <4 x float> %[[RET]]
    unsafe { transmute(v) }
}

// CHECK-LABEL: <2 x ptr> @float_ptr_same_lanes(<2 x double> %v)
#[no_mangle]
pub extern "C" fn float_ptr_same_lanes(v: F64X2) -> PtrX2 {
    // CHECK-NOT: alloca
    // CHECK: %[[TEMP:.+]] = alloca [16 x i8]
    // CHECK-NOT: alloca
    // CHECK: call void @llvm.lifetime.start.p0(i64 16, ptr %[[TEMP]])
    // CHECK: store <2 x double> %v, ptr %[[TEMP]]
    // CHECK: %[[RET:.+]] = load <2 x ptr>, ptr %[[TEMP]]
    // CHECK: call void @llvm.lifetime.end.p0(i64 16, ptr %[[TEMP]])
    // CHECK: ret <2 x ptr> %[[RET]]
    unsafe { transmute(v) }
}

// CHECK-LABEL: <2 x double> @ptr_float_same_lanes(<2 x ptr> %v)
#[no_mangle]
pub extern "C" fn ptr_float_same_lanes(v: PtrX2) -> F64X2 {
    // CHECK-NOT: alloca
    // CHECK: %[[TEMP:.+]] = alloca [16 x i8]
    // CHECK-NOT: alloca
    // CHECK: call void @llvm.lifetime.start.p0(i64 16, ptr %[[TEMP]])
    // CHECK: store <2 x ptr> %v, ptr %[[TEMP]]
    // CHECK: %[[RET:.+]] = load <2 x double>, ptr %[[TEMP]]
    // CHECK: call void @llvm.lifetime.end.p0(i64 16, ptr %[[TEMP]])
    // CHECK: ret <2 x double> %[[RET]]
    unsafe { transmute(v) }
}

// CHECK-LABEL: <2 x ptr> @int_ptr_same_lanes(<2 x i64> %v)
#[no_mangle]
pub extern "C" fn int_ptr_same_lanes(v: I64X2) -> PtrX2 {
    // CHECK-NOT: alloca
    // CHECK: %[[TEMP:.+]] = alloca [16 x i8]
    // CHECK-NOT: alloca
    // CHECK: call void @llvm.lifetime.start.p0(i64 16, ptr %[[TEMP]])
    // CHECK: store <2 x i64> %v, ptr %[[TEMP]]
    // CHECK: %[[RET:.+]] = load <2 x ptr>, ptr %[[TEMP]]
    // CHECK: call void @llvm.lifetime.end.p0(i64 16, ptr %[[TEMP]])
    // CHECK: ret <2 x ptr> %[[RET]]
    unsafe { transmute(v) }
}

// CHECK-LABEL: <2 x i64> @ptr_int_same_lanes(<2 x ptr> %v)
#[no_mangle]
pub extern "C" fn ptr_int_same_lanes(v: PtrX2) -> I64X2 {
    // CHECK-NOT: alloca
    // CHECK: %[[TEMP:.+]] = alloca [16 x i8]
    // CHECK-NOT: alloca
    // CHECK: call void @llvm.lifetime.start.p0(i64 16, ptr %[[TEMP]])
    // CHECK: store <2 x ptr> %v, ptr %[[TEMP]]
    // CHECK: %[[RET:.+]] = load <2 x i64>, ptr %[[TEMP]]
    // CHECK: call void @llvm.lifetime.end.p0(i64 16, ptr %[[TEMP]])
    // CHECK: ret <2 x i64> %[[RET]]
    unsafe { transmute(v) }
}

// CHECK-LABEL: <2 x ptr> @float_ptr_widen(<4 x float> %v)
#[no_mangle]
pub extern "C" fn float_ptr_widen(v: F32X4) -> PtrX2 {
    // CHECK-NOT: alloca
    // CHECK: %[[TEMP:.+]] = alloca [16 x i8]
    // CHECK-NOT: alloca
    // CHECK: call void @llvm.lifetime.start.p0(i64 16, ptr %[[TEMP]])
    // CHECK: store <4 x float> %v, ptr %[[TEMP]]
    // CHECK: %[[RET:.+]] = load <2 x ptr>, ptr %[[TEMP]]
    // CHECK: call void @llvm.lifetime.end.p0(i64 16, ptr %[[TEMP]])
    // CHECK: ret <2 x ptr> %[[RET]]
    unsafe { transmute(v) }
}

// CHECK-LABEL: <2 x ptr> @int_ptr_widen(<4 x i32> %v)
#[no_mangle]
pub extern "C" fn int_ptr_widen(v: I32X4) -> PtrX2 {
    // CHECK-NOT: alloca
    // CHECK: %[[TEMP:.+]] = alloca [16 x i8]
    // CHECK-NOT: alloca
    // CHECK: call void @llvm.lifetime.start.p0(i64 16, ptr %[[TEMP]])
    // CHECK: store <4 x i32> %v, ptr %[[TEMP]]
    // CHECK: %[[RET:.+]] = load <2 x ptr>, ptr %[[TEMP]]
    // CHECK: call void @llvm.lifetime.end.p0(i64 16, ptr %[[TEMP]])
    // CHECK: ret <2 x ptr> %[[RET]]
    unsafe { transmute(v) }
}
