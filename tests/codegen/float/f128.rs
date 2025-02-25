// 32-bit x86 returns float types differently to avoid the x87 stack.
// 32-bit systems will return 128bit values using a return area pointer.
// Emscripten aligns f128 to 8 bytes, not 16.
//@ revisions: x86-sse x86-nosse bit32 bit64 emscripten
//@[x86-sse] only-x86
//@[x86-sse] only-rustc_abi-x86-sse2
//@[x86-nosse] only-x86
//@[x86-nosse] ignore-rustc_abi-x86-sse2
//@[bit32] ignore-x86
//@[bit32] ignore-emscripten
//@[bit32] only-32bit
//@[bit64] ignore-x86
//@[bit64] ignore-emscripten
//@[bit64] only-64bit
//@[emscripten] only-emscripten

// Verify that our intrinsics generate the correct LLVM calls for f128

#![crate_type = "lib"]
#![feature(f128)]
#![feature(f16)]
#![feature(core_intrinsics)]

// CHECK-LABEL: i1 @f128_eq(
#[no_mangle]
pub fn f128_eq(a: f128, b: f128) -> bool {
    // CHECK: fcmp oeq fp128 %{{.+}}, %{{.+}}
    a == b
}

// CHECK-LABEL: i1 @f128_ne(
#[no_mangle]
pub fn f128_ne(a: f128, b: f128) -> bool {
    // CHECK: fcmp une fp128 %{{.+}}, %{{.+}}
    a != b
}

// CHECK-LABEL: i1 @f128_gt(
#[no_mangle]
pub fn f128_gt(a: f128, b: f128) -> bool {
    // CHECK: fcmp ogt fp128 %{{.+}}, %{{.+}}
    a > b
}

// CHECK-LABEL: i1 @f128_ge(
#[no_mangle]
pub fn f128_ge(a: f128, b: f128) -> bool {
    // CHECK: fcmp oge fp128 %{{.+}}, %{{.+}}
    a >= b
}

// CHECK-LABEL: i1 @f128_lt(
#[no_mangle]
pub fn f128_lt(a: f128, b: f128) -> bool {
    // CHECK: fcmp olt fp128 %{{.+}}, %{{.+}}
    a < b
}

// CHECK-LABEL: i1 @f128_le(
#[no_mangle]
pub fn f128_le(a: f128, b: f128) -> bool {
    // CHECK: fcmp ole fp128 %{{.+}}, %{{.+}}
    a <= b
}

// x86-nosse-LABEL: void @f128_neg({{.*}}sret([16 x i8])
// x86-sse-LABEL: <16 x i8> @f128_neg(fp128
// bit32-LABEL: void @f128_neg({{.*}}sret([16 x i8])
// bit64-LABEL: fp128 @f128_neg(
// emscripten-LABEL: void @f128_neg({{.*}}sret([16 x i8])
#[no_mangle]
pub fn f128_neg(a: f128) -> f128 {
    // CHECK: fneg fp128
    -a
}

// x86-nosse-LABEL: void @f128_add({{.*}}sret([16 x i8])
// x86-sse-LABEL: <16 x i8> @f128_add(fp128
// bit32-LABEL: void @f128_add({{.*}}sret([16 x i8])
// bit64-LABEL: fp128 @f128_add(
// emscripten-LABEL: void @f128_add({{.*}}sret([16 x i8])
#[no_mangle]
pub fn f128_add(a: f128, b: f128) -> f128 {
    // CHECK: fadd fp128 %{{.+}}, %{{.+}}
    a + b
}

// x86-nosse-LABEL: void @f128_sub({{.*}}sret([16 x i8])
// x86-sse-LABEL: <16 x i8> @f128_sub(fp128
// bit32-LABEL: void @f128_sub({{.*}}sret([16 x i8])
// bit64-LABEL: fp128 @f128_sub(
// emscripten-LABEL: void @f128_sub({{.*}}sret([16 x i8])
#[no_mangle]
pub fn f128_sub(a: f128, b: f128) -> f128 {
    // CHECK: fsub fp128 %{{.+}}, %{{.+}}
    a - b
}

// x86-nosse-LABEL: void @f128_mul({{.*}}sret([16 x i8])
// x86-sse-LABEL: <16 x i8> @f128_mul(fp128
// bit32-LABEL: void @f128_mul({{.*}}sret([16 x i8])
// bit64-LABEL: fp128 @f128_mul(
// emscripten-LABEL: void @f128_mul({{.*}}sret([16 x i8])
#[no_mangle]
pub fn f128_mul(a: f128, b: f128) -> f128 {
    // CHECK: fmul fp128 %{{.+}}, %{{.+}}
    a * b
}

// x86-nosse-LABEL: void @f128_div({{.*}}sret([16 x i8])
// x86-sse-LABEL: <16 x i8> @f128_div(fp128
// bit32-LABEL: void @f128_div({{.*}}sret([16 x i8])
// bit64-LABEL: fp128 @f128_div(
// emscripten-LABEL: void @f128_div({{.*}}sret([16 x i8])
#[no_mangle]
pub fn f128_div(a: f128, b: f128) -> f128 {
    // CHECK: fdiv fp128 %{{.+}}, %{{.+}}
    a / b
}

// x86-nosse-LABEL: void @f128_rem({{.*}}sret([16 x i8])
// x86-sse-LABEL: <16 x i8> @f128_rem(fp128
// bit32-LABEL: void @f128_rem({{.*}}sret([16 x i8])
// bit64-LABEL: fp128 @f128_rem(
// emscripten-LABEL: void @f128_rem({{.*}}sret([16 x i8])
#[no_mangle]
pub fn f128_rem(a: f128, b: f128) -> f128 {
    // CHECK: frem fp128 %{{.+}}, %{{.+}}
    a % b
}

// CHECK-LABEL: void @f128_add_assign(
#[no_mangle]
pub fn f128_add_assign(a: &mut f128, b: f128) {
    // CHECK: fadd fp128 %{{.+}}, %{{.+}}
    // CHECK-NEXT: store fp128 %{{.+}}, ptr %{{.+}}
    *a += b;
}

// CHECK-LABEL: void @f128_sub_assign(
#[no_mangle]
pub fn f128_sub_assign(a: &mut f128, b: f128) {
    // CHECK: fsub fp128 %{{.+}}, %{{.+}}
    // CHECK-NEXT: store fp128 %{{.+}}, ptr %{{.+}}
    *a -= b;
}

// CHECK-LABEL: void @f128_mul_assign(
#[no_mangle]
pub fn f128_mul_assign(a: &mut f128, b: f128) {
    // CHECK: fmul fp128 %{{.+}}, %{{.+}}
    // CHECK-NEXT: store fp128 %{{.+}}, ptr %{{.+}}
    *a *= b
}

// CHECK-LABEL: void @f128_div_assign(
#[no_mangle]
pub fn f128_div_assign(a: &mut f128, b: f128) {
    // CHECK: fdiv fp128 %{{.+}}, %{{.+}}
    // CHECK-NEXT: store fp128 %{{.+}}, ptr %{{.+}}
    *a /= b
}

// CHECK-LABEL: void @f128_rem_assign(
#[no_mangle]
pub fn f128_rem_assign(a: &mut f128, b: f128) {
    // CHECK: frem fp128 %{{.+}}, %{{.+}}
    // CHECK-NEXT: store fp128 %{{.+}}, ptr %{{.+}}
    *a %= b
}

/* float to float conversions */

// x86-sse-LABEL: <2 x i8> @f128_as_f16(
// x86-nosse-LABEL: i16 @f128_as_f16(
// bits32-LABEL: half @f128_as_f16(
// bits64-LABEL: half @f128_as_f16(
#[no_mangle]
pub fn f128_as_f16(a: f128) -> f16 {
    // CHECK: fptrunc fp128 %{{.+}} to half
    a as f16
}

// x86-sse-LABEL: <4 x i8> @f128_as_f32(
// x86-nosse-LABEL: i32 @f128_as_f32(
// bit32-LABEL: float @f128_as_f32(
// bit64-LABEL: float @f128_as_f32(
// emscripten-LABEL: float @f128_as_f32(
#[no_mangle]
pub fn f128_as_f32(a: f128) -> f32 {
    // CHECK: fptrunc fp128 %{{.+}} to float
    a as f32
}

// x86-sse-LABEL: <8 x i8> @f128_as_f64(
// x86-nosse-LABEL: void @f128_as_f64({{.*}}sret([8 x i8])
// bit32-LABEL: double @f128_as_f64(
// bit64-LABEL: double @f128_as_f64(
// emscripten-LABEL: double @f128_as_f64(
#[no_mangle]
pub fn f128_as_f64(a: f128) -> f64 {
    // CHECK: fptrunc fp128 %{{.+}} to double
    a as f64
}

// x86-sse-LABEL: <16 x i8> @f128_as_self(
// x86-nosse-LABEL: void @f128_as_self({{.*}}sret([16 x i8])
// bit32-LABEL: void @f128_as_self({{.*}}sret([16 x i8])
// bit64-LABEL: fp128 @f128_as_self(
// emscripten-LABEL: void @f128_as_self({{.*}}sret([16 x i8])
#[no_mangle]
pub fn f128_as_self(a: f128) -> f128 {
    // x86: store fp128 %a, ptr %_0, align 16
    // bit32: store fp128 %a, ptr %_0, align 16
    // bit64: ret fp128 %{{.+}}
    // emscripten: store fp128 %a, ptr %_0, align 8
    a as f128
}

// x86-sse-LABEL: <16 x i8> @f16_as_f128(
// x86-nosse-LABEL: void @f16_as_f128({{.*}}sret([16 x i8])
// bit32-LABEL: void @f16_as_f128({{.*}}sret([16 x i8])
// bit64-LABEL: fp128 @f16_as_f128(
// emscripten-LABEL: void @f16_as_f128({{.*}}sret([16 x i8])
#[no_mangle]
pub fn f16_as_f128(a: f16) -> f128 {
    // CHECK: fpext half %{{.+}} to fp128
    a as f128
}

// x86-sse-LABEL: <16 x i8> @f32_as_f128(
// x86-nosse-LABEL: void @f32_as_f128({{.*}}sret([16 x i8])
// bit32-LABEL: void @f32_as_f128({{.*}}sret([16 x i8])
// bit64-LABEL: fp128 @f32_as_f128(
// emscripten-LABEL: void @f32_as_f128({{.*}}sret([16 x i8])
#[no_mangle]
pub fn f32_as_f128(a: f32) -> f128 {
    // CHECK: fpext float %{{.+}} to fp128
    a as f128
}

// x86-sse-LABEL: <16 x i8> @f64_as_f128(
// x86-nosse-LABEL: void @f64_as_f128({{.*}}sret([16 x i8])
// bit32-LABEL: void @f64_as_f128({{.*}}sret([16 x i8])
// bit64-LABEL: fp128 @f64_as_f128(
// emscripten-LABEL: void @f64_as_f128({{.*}}sret([16 x i8])
#[no_mangle]
pub fn f64_as_f128(a: f64) -> f128 {
    // CHECK: fpext double %{{.+}} to fp128
    a as f128
}

/* float to int conversions */

// CHECK-LABEL: i8 @f128_as_u8(
#[no_mangle]
pub fn f128_as_u8(a: f128) -> u8 {
    // CHECK: call i8 @llvm.fptoui.sat.i8.f128(fp128 %{{.+}})
    a as u8
}

#[no_mangle]
pub fn f128_as_u16(a: f128) -> u16 {
    // CHECK: call i16 @llvm.fptoui.sat.i16.f128(fp128 %{{.+}})
    a as u16
}

// CHECK-LABEL: i32 @f128_as_u32(
#[no_mangle]
pub fn f128_as_u32(a: f128) -> u32 {
    // CHECK: call i32 @llvm.fptoui.sat.i32.f128(fp128 %{{.+}})
    a as u32
}

// CHECK-LABEL: i64 @f128_as_u64(
#[no_mangle]
pub fn f128_as_u64(a: f128) -> u64 {
    // CHECK: call i64 @llvm.fptoui.sat.i64.f128(fp128 %{{.+}})
    a as u64
}

// x86-sse-LABEL: void @f128_as_u128({{.*}}sret([16 x i8])
// x86-nosse-LABEL: void @f128_as_u128({{.*}}sret([16 x i8])
// bit32-LABEL: void @f128_as_u128({{.*}}sret([16 x i8])
// bit64-LABEL: i128 @f128_as_u128(
// emscripten-LABEL: void @f128_as_u128({{.*}}sret([16 x i8])
#[no_mangle]
pub fn f128_as_u128(a: f128) -> u128 {
    // CHECK: call i128 @llvm.fptoui.sat.i128.f128(fp128 %{{.+}})
    a as u128
}

// CHECK-LABEL: i8 @f128_as_i8(
#[no_mangle]
pub fn f128_as_i8(a: f128) -> i8 {
    // CHECK: call i8 @llvm.fptosi.sat.i8.f128(fp128 %{{.+}})
    a as i8
}

// CHECK-LABEL: i16 @f128_as_i16(
#[no_mangle]
pub fn f128_as_i16(a: f128) -> i16 {
    // CHECK: call i16 @llvm.fptosi.sat.i16.f128(fp128 %{{.+}})
    a as i16
}
// CHECK-LABEL: i32 @f128_as_i32(
#[no_mangle]
pub fn f128_as_i32(a: f128) -> i32 {
    // CHECK: call i32 @llvm.fptosi.sat.i32.f128(fp128 %{{.+}})
    a as i32
}

// CHECK-LABEL: i64 @f128_as_i64(
#[no_mangle]
pub fn f128_as_i64(a: f128) -> i64 {
    // CHECK: call i64 @llvm.fptosi.sat.i64.f128(fp128 %{{.+}})
    a as i64
}

// x86-sse-LABEL: void @f128_as_i128({{.*}}sret([16 x i8])
// x86-nosse-LABEL: void @f128_as_i128({{.*}}sret([16 x i8])
// bit32-LABEL: void @f128_as_i128({{.*}}sret([16 x i8])
// bit64-LABEL: i128 @f128_as_i128(
// emscripten-LABEL: void @f128_as_i128({{.*}}sret([16 x i8])
#[no_mangle]
pub fn f128_as_i128(a: f128) -> i128 {
    // CHECK: call i128 @llvm.fptosi.sat.i128.f128(fp128 %{{.+}})
    a as i128
}

/* int to float conversions */

// x86-sse-LABEL: <16 x i8> @u8_as_f128(
// x86-nosse-LABEL: void @u8_as_f128({{.*}}sret([16 x i8])
// bit32-LABEL: void @u8_as_f128({{.*}}sret([16 x i8])
// bit64-LABEL: fp128 @u8_as_f128(
// emscripten-LABEL: void @u8_as_f128({{.*}}sret([16 x i8])
#[no_mangle]
pub fn u8_as_f128(a: u8) -> f128 {
    // CHECK: uitofp i8 %{{.+}} to fp128
    a as f128
}

// x86-sse-LABEL: <16 x i8> @u16_as_f128(
// x86-nosse-LABEL: void @u16_as_f128({{.*}}sret([16 x i8])
// bit32-LABEL: void @u16_as_f128({{.*}}sret([16 x i8])
// bit64-LABEL: fp128 @u16_as_f128(
// emscripten-LABEL: void @u16_as_f128({{.*}}sret([16 x i8])
#[no_mangle]
pub fn u16_as_f128(a: u16) -> f128 {
    // CHECK: uitofp i16 %{{.+}} to fp128
    a as f128
}

// x86-sse-LABEL: <16 x i8> @u32_as_f128(
// x86-nosse-LABEL: void @u32_as_f128({{.*}}sret([16 x i8])
// bit32-LABEL: void @u32_as_f128({{.*}}sret([16 x i8])
// bit64-LABEL: fp128 @u32_as_f128(
// emscripten-LABEL: void @u32_as_f128({{.*}}sret([16 x i8])
#[no_mangle]
pub fn u32_as_f128(a: u32) -> f128 {
    // CHECK: uitofp i32 %{{.+}} to fp128
    a as f128
}

// x86-sse-LABEL: <16 x i8> @u64_as_f128(
// x86-nosse-LABEL: void @u64_as_f128({{.*}}sret([16 x i8])
// bit32-LABEL: void @u64_as_f128({{.*}}sret([16 x i8])
// bit64-LABEL: fp128 @u64_as_f128(
// emscripten-LABEL: void @u64_as_f128({{.*}}sret([16 x i8])
#[no_mangle]
pub fn u64_as_f128(a: u64) -> f128 {
    // CHECK: uitofp i64 %{{.+}} to fp128
    a as f128
}

// x86-sse-LABEL: <16 x i8> @u128_as_f128(
// x86-nosse-LABEL: void @u128_as_f128({{.*}}sret([16 x i8])
// bit32-LABEL: void @u128_as_f128({{.*}}sret([16 x i8])
// bit64-LABEL: fp128 @u128_as_f128(
// emscripten-LABEL: void @u128_as_f128({{.*}}sret([16 x i8])
#[no_mangle]
pub fn u128_as_f128(a: u128) -> f128 {
    // CHECK: uitofp i128 %{{.+}} to fp128
    a as f128
}

// x86-sse-LABEL: <16 x i8> @i8_as_f128(
// x86-nosse-LABEL: void @i8_as_f128({{.*}}sret([16 x i8])
// bit32-LABEL: void @i8_as_f128({{.*}}sret([16 x i8])
// bit64-LABEL: fp128 @i8_as_f128(
// emscripten-LABEL: void @i8_as_f128({{.*}}sret([16 x i8])
#[no_mangle]
pub fn i8_as_f128(a: i8) -> f128 {
    // CHECK: sitofp i8 %{{.+}} to fp128
    a as f128
}

// x86-sse-LABEL: <16 x i8> @i16_as_f128(
// x86-nosse-LABEL: void @i16_as_f128({{.*}}sret([16 x i8])
// bit32-LABEL: void @i16_as_f128({{.*}}sret([16 x i8])
// bit64-LABEL: fp128 @i16_as_f128(
// emscripten-LABEL: void @i16_as_f128({{.*}}sret([16 x i8])
#[no_mangle]
pub fn i16_as_f128(a: i16) -> f128 {
    // CHECK: sitofp i16 %{{.+}} to fp128
    a as f128
}

// x86-sse-LABEL: <16 x i8> @i32_as_f128(
// x86-nosse-LABEL: void @i32_as_f128({{.*}}sret([16 x i8])
// bit32-LABEL: void @i32_as_f128({{.*}}sret([16 x i8])
// bit64-LABEL: fp128 @i32_as_f128(
// emscripten-LABEL: void @i32_as_f128({{.*}}sret([16 x i8])
#[no_mangle]
pub fn i32_as_f128(a: i32) -> f128 {
    // CHECK: sitofp i32 %{{.+}} to fp128
    a as f128
}

// x86-sse-LABEL: <16 x i8> @i64_as_f128(
// x86-nosse-LABEL: void @i64_as_f128({{.*}}sret([16 x i8])
// bit32-LABEL: void @i64_as_f128({{.*}}sret([16 x i8])
// bit64-LABEL: fp128 @i64_as_f128(
// emscripten-LABEL: void @i64_as_f128({{.*}}sret([16 x i8])
#[no_mangle]
pub fn i64_as_f128(a: i64) -> f128 {
    // CHECK: sitofp i64 %{{.+}} to fp128
    a as f128
}

// x86-sse-LABEL: <16 x i8> @i128_as_f128(
// x86-nosse-LABEL: void @i128_as_f128({{.*}}sret([16 x i8])
// bit32-LABEL: void @i128_as_f128({{.*}}sret([16 x i8])
// bit64-LABEL: fp128 @i128_as_f128(
// emscripten-LABEL: void @i128_as_f128({{.*}}sret([16 x i8])
#[no_mangle]
pub fn i128_as_f128(a: i128) -> f128 {
    // CHECK: sitofp i128 %{{.+}} to fp128
    a as f128
}
