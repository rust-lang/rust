// ignore-tidy-file-linelength
//@ add-minicore
// Run it at O0, so that the compiler doesn't optimise the calls away.
//@ revisions: DISC NO_DISC
//@ [DISC] needs-llvm-components: aarch64
//@ [DISC] compile-flags: --target=aarch64-unknown-linux-pauthtest --crate-type=lib -Zpointer-authentication=+function-pointer-type-discrimination -C opt-level=0
//@ [NO_DISC] needs-llvm-components: aarch64
//@ [NO_DISC] compile-flags: --target=aarch64-unknown-linux-pauthtest --crate-type=lib -Zpointer-authentication=-function-pointer-type-discrimination -C opt-level=0
//
// Function pointer type discriminators for `core::num::Complex<T>` must match Clang's encoding of C
// `_Complex T`: 'C' followed by the element type.

#![feature(no_core, lang_items, f16, f128)]
#![no_std]
#![no_core]
#![crate_type = "lib"]

extern crate minicore;
use minicore::num::Complex;

#[repr(transparent)]
pub struct Wrap(Complex<f32>);

// An ordinary repr(C) struct that happens to be named `Complex`: not `_Complex`.
pub mod user {
    #[repr(C)]
    pub struct Complex {
        pub re: f32,
        pub im: f32,
    }
}

// C: void (*)(float _Complex) -> "FvCfE"
// CHECK-LABEL: @call_cf
#[no_mangle]
pub fn call_cf(f: extern "C" fn(Complex<f32>), x: Complex<f32>) {
    // DISC: [ "ptrauth"(i32 0, i64 5745) ]
    // NO_DISC: [ "ptrauth"(i32 0, i64 0) ]
    f(x)
}

// C: void (*)(double _Complex) -> "FvCdE"
// CHECK-LABEL: @call_cd
#[no_mangle]
pub fn call_cd(f: extern "C" fn(Complex<f64>), x: Complex<f64>) {
    // DISC: [ "ptrauth"(i32 0, i64 65236) ]
    // NO_DISC: [ "ptrauth"(i32 0, i64 0) ]
    f(x)
}

// C: void (*)(int _Complex) -> "FvCiE"
// CHECK-LABEL: @call_ci
// DISC: [ "ptrauth"(i32 0, i64 13305) ]
// NO_DISC: [ "ptrauth"(i32 0, i64 0) ]
#[no_mangle]
pub fn call_ci(f: extern "C" fn(Complex<i32>), x: Complex<i32>) {
    f(x)
}

// C: long double _Complex (*)(void) -> "FCeE" (long double is binary128 here)
// CHECK-LABEL: @call_ret_ce
#[no_mangle]
pub fn call_ret_ce(f: extern "C" fn() -> Complex<f128>) -> Complex<f128> {
    // DISC: [ "ptrauth"(i32 0, i64 776) ]
    // NO_DISC: [ "ptrauth"(i32 0, i64 0) ]
    f()
}

// C: _Float16 _Complex (*)(void) -> "FCDF16_E"
// CHECK-LABEL: @call_ret_ch
#[no_mangle]
pub fn call_ret_ch(f: extern "C" fn() -> Complex<f16>) -> Complex<f16> {
    // DISC: [ "ptrauth"(i32 0, i64 17465) ]
    // NO_DISC: [ "ptrauth"(i32 0, i64 0) ]
    f()
}

// C: double _Complex (*)(float _Complex) -> "FCdCfE"
// CHECK-LABEL: @call_cf_to_cd
#[no_mangle]
pub fn call_cf_to_cd(
    f: extern "C" fn(Complex<f32>) -> Complex<f64>,
    x: Complex<f32>,
) -> Complex<f64> {
    // DISC: [ "ptrauth"(i32 0, i64 24171) ]
    // NO_DISC: [ "ptrauth"(i32 0, i64 0) ]
    f(x)
}

// C: void (*)(float _Complex, int _Complex, double _Complex) -> "FvCfCiCdE"
// CHECK-LABEL: @call_mixed
#[no_mangle]
pub fn call_mixed(
    f: extern "C" fn(Complex<f32>, Complex<i32>, Complex<f64>),
    a: Complex<f32>,
    b: Complex<i32>,
    c: Complex<f64>,
) {
    // DISC: [ "ptrauth"(i32 0, i64 44841) ]
    // NO_DISC: [ "ptrauth"(i32 0, i64 0) ]
    f(a, b, c)
}

// A repr(transparent) wrapper is canonicalized away: same as `call_cf`.
// CHECK-LABEL: @call_wrap
#[no_mangle]
pub fn call_wrap(f: extern "C" fn(Wrap), x: Wrap) {
    // DISC: [ "ptrauth"(i32 0, i64 5745) ]
    // NO_DISC: [ "ptrauth"(i32 0, i64 0) ]
    f(x)
}

// C: struct Complex { float re, im; }; void (*)(struct Complex) -> "Fv7ComplexE"
// CHECK-LABEL: @call_user
#[no_mangle]
pub fn call_user(f: extern "C" fn(user::Complex), x: user::Complex) {
    // DISC: [ "ptrauth"(i32 0, i64 50183) ]
    // NO_DISC: [ "ptrauth"(i32 0, i64 0) ]
    f(x)
}

// Baseline float, not complex. C: void (*)(float) -> "FvfE"
// CHECK-LABEL: @call_f
#[no_mangle]
pub fn call_f(f: extern "C" fn(f32), x: f32) {
    // DISC: [ "ptrauth"(i32 0, i64 21613) ]
    // NO_DISC: [ "ptrauth"(i32 0, i64 0) ]
    f(x)
}

// Baseline two floats, still not complex. C: void (*)(float, float) -> "FvffE"
// CHECK-LABEL: @call_ff
#[no_mangle]
pub fn call_ff(f: extern "C" fn(f32, f32), a: f32, b: f32) {
    // DISC: [ "ptrauth"(i32 0, i64 47857) ]
    // NO_DISC: [ "ptrauth"(i32 0, i64 0) ]
    f(a, b)
}
