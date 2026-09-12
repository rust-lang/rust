// Tests that MIPS targets return structs that are up to 128 bits large, only
// consist of 1 or 2 floating point fields and have a first field with offset 0
// in FPRs rather than GPRs.
//
//@ add-minicore
//@ assembly-output: emit-asm
//@ compile-flags: -Copt-level=3
//
//@ revisions: LE
//@[LE] compile-flags: --target=mips64el-unknown-linux-gnuabi64
//@[LE] needs-llvm-components: mips
//@ revisions: BE
//@[BE] compile-flags: --target=mips64-unknown-linux-gnuabi64
//@[BE] needs-llvm-components: mips

#![crate_type = "lib"]
#![feature(no_core, f128)]
#![no_core]

extern crate minicore;

// 128-bit large struct with one floating point field, should be returned in
// $f0 and $f1 to match the de-facto GCC ABI.
#[repr(C)]
struct SingleF128 {
    x: f128,
}

// 64-bit large struct with one floating point field, should be returned in $f0.
#[repr(C)]
struct SingleF64 {
    x: f64,
}

// 32-bit large struct with one floating point field, should be returned in $f0.
#[repr(C)]
struct SingleF32 {
    x: f32,
}

// 128-bit large struct with two floating point fields, should be returned in
// $f0 and $f2.
#[repr(C)]
struct TwoF64 {
    x: f64,
    y: f64,
}

// 64-bit large struct with two floating point fields, should be returned in
// $f0 and $f2.
#[repr(C)]
struct TwoF32 {
    x: f32,
    y: f32,
}

// 96-bit large struct with two floating point fields, should be returned in
// $f0 and $f2.
#[repr(C)]
struct F32AndF64 {
    x: f32,
    y: f64,
}

// 96-bit large struct with two floating-point fields, should be returned in
// $f0 and $f2.
#[repr(C)]
struct F64AndF32 {
    x: f64,
    y: f32,
}

// CHECK-LABEL: single_f128
// CHECK: dmtc1 $4, $f0
// CHECK-NEXT: jr $ra
// CHECK-NEXT: dmtc1 $5, $f1
#[unsafe(no_mangle)]
pub extern "C" fn single_f128(a: SingleF128) -> SingleF128 {
    a
}

// CHECK-LABEL: single_f64
// CHECK: jr $ra
// CHECK-NEXT: mov.d $f0, $f12
#[unsafe(no_mangle)]
pub extern "C" fn single_f64(a: SingleF64) -> SingleF64 {
    a
}

// CHECK-LABEL: single_f32
// BE: dsrl $1, $4, 32
// LE: sll $1, $4, 0
// BE-NEXT: sll $1, $1, 0
// CHECK-NEXT: jr $ra
// CHECK-NEXT: mtc1 $1, $f0
#[unsafe(no_mangle)]
pub extern "C" fn single_f32(a: SingleF32) -> SingleF32 {
    a
}

// CHECK-LABEL: two_f64
// CHECK: mov.d $f2, $f13
// CHECK-NEXT: jr $ra
// CHECK-NEXT: mov.d $f0, $f12
#[unsafe(no_mangle)]
pub extern "C" fn two_f64(a: TwoF64) -> TwoF64 {
    a
}

// CHECK-LABEL: two_f32
// CHECK: sll $1, $4, 0
// LE-NEXT: mtc1 $1, $f0
// BE-NEXT: mtc1 $1, $f2
// CHECK-NEXT: dsrl $1, $4, 32
// CHECK-NEXT: sll $1, $1, 0
// CHECK-NEXT: jr $ra
// LE-NEXT: mtc1 $1, $f2
// BE-NEXT: mtc1 $1, $f0
#[unsafe(no_mangle)]
pub extern "C" fn two_f32(a: TwoF32) -> TwoF32 {
    a
}

// We deviate from Clang in the order of instructions in the next two functions
// but that's okay

// CHECK-LABEL: f32_and_f64
// BE: dsrl $1, $4, 32
// LE: mov.d $f2, $f13
// BE-NEXT: mov.d $f2, $f13
// LE-NEXT: sll $1, $4, 0
// BE-NEXT: sll $1, $1, 0
// CHECK-NEXT: jr $ra
// CHECK-NEXT: mtc1 $1, $f0
#[unsafe(no_mangle)]
pub extern "C" fn f32_and_f64(a: F32AndF64) -> F32AndF64 {
    a
}

// CHECK-LABEL: f64_and_f32
// BE: dsrl $1, $5, 32
// LE: mov.d $f0, $f12
// BE-NEXT: mov.d $f0, $f12
// LE-NEXT: sll $1, $5, 0
// BE-NEXT: sll $1, $1, 0
// CHECK-NEXT: jr $ra
// CHECK-NEXT: mtc1 $1, $f2
#[unsafe(no_mangle)]
pub extern "C" fn f64_and_f32(a: F64AndF32) -> F64AndF32 {
    a
}
