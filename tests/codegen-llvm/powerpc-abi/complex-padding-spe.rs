// ignore-tidy-file-linelength
//@ add-minicore
//@ compile-flags: -C no-prepopulate-passes -Z codegen-source-order -C opt-level=3

//@ revisions: FLT SPE
//@ [FLT] compile-flags: --target powerpc-unknown-linux-gnu
//@ [FLT] needs-llvm-components: powerpc
//@ [SPE] compile-flags: --target powerpc-unknown-linux-gnuspe
//@ [SPE] needs-llvm-components: powerpc

// Checks that on 32-bit powerpc with SPE a `Complex<T>` is never split between the GPRs and the
// stack. SPE is a softfloat target that passes `f32`, `f64` and `ppcf128` in GPRs, with an `f64`
// in an even-odd GPR pair.

#![feature(no_core, lang_items)]
#![no_core]
#![expect(improper_ctypes_definitions)]
#![crate_type = "lib"]

extern crate minicore;
use minicore::arch::powerpc::ppcf128;
use minicore::num::Complex;
use minicore::*;

type Cld = Complex<ppcf128>;

// FLT: define{{.*}} void @float(float noundef %a, [8 x i32] %0)
// SPE: define{{.*}} void @float(float noundef %a, i32 %0, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, [8 x i32] %7)
#[unsafe(no_mangle)]
pub extern "C" fn float(a: f32, x: Cld) {}

// FLT: define{{.*}} void @double(double noundef %a, [8 x i32] %0)
// SPE: define{{.*}} void @double(double noundef %a, i32 %0, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, [8 x i32] %6)
#[unsafe(no_mangle)]
pub extern "C" fn double(a: f64, x: Cld) {}

// With SPE the `f64` is in r5..=r6, r4 is skipped.
//
// FLT: define{{.*}} void @double_aligned(float noundef %a, double noundef %b, [8 x i32] %0)
// SPE: define{{.*}} void @double_aligned(float noundef %a, double noundef %b, i32 %0, i32 %1, i32 %2, i32 %3, [8 x i32] %4)
#[unsafe(no_mangle)]
pub extern "C" fn double_aligned(a: f32, b: f64, x: Cld) {}

// FLT: define{{.*}} void @long_double(ppc_fp128 noundef %a, [8 x i32] %0)
// SPE: define{{.*}} void @long_double(ppc_fp128 noundef %a, i32 %0, i32 %1, i32 %2, i32 %3, [8 x i32] %4)
#[unsafe(no_mangle)]
pub extern "C" fn long_double(a: ppcf128, x: Cld) {}
