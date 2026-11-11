// ignore-tidy-file-linelength
//@ add-minicore
//@ compile-flags: -C no-prepopulate-passes -Z codegen-source-order -C opt-level=3

//@ revisions: POWERPC ALTIVEC VSX
//@ [POWERPC] compile-flags: --target powerpc-unknown-linux-gnu
//@ [POWERPC] needs-llvm-components: powerpc
//@ [ALTIVEC] compile-flags: --target powerpc-unknown-linux-gnu -C target-feature=+altivec
//@ [ALTIVEC] needs-llvm-components: powerpc
//@ [VSX] compile-flags: --target powerpc-unknown-linux-gnu -C target-feature=+vsx
//@ [VSX] needs-llvm-components: powerpc

// Checks that on 32-bit powerpc a `Complex<T>` is never split between the GPRs and the stack.
// When a complex number only partially fits in the remaining registers, the complex value and
// any subsequent arguments are passed via the stack.
//
// To know whether an argument fits, we need an accurate accounting of how many GPRs are
// already used. This file tests that logic.
//
// The padding arguments are not marked as `noundef`, and usually have names like %0, %1, etc.

#![feature(no_core, lang_items, f16, f128)]
#![no_core]
#![expect(improper_ctypes_definitions)]
#![crate_type = "lib"]

extern crate minicore;
use minicore::arch::powerpc::ppcf128;
use minicore::num::Complex;
use minicore::simd::*;
use minicore::*;

type Cf = Complex<f32>;
type Cd = Complex<f64>;
type Cld = Complex<ppcf128>;

/// A type big enough to be returned via sret pointer, and hence claim a GPR.
#[repr(C)]
pub struct Big([u32; 8]);

/// A zero-sized type, which is passed by reference, claiming a GPR.
#[repr(C)]
pub struct Empty {}

// The complex value fits in the remaining GPRs, no padding needed.

// CHECK: define{{.*}} void @fits_last(i32 noundef %r3, i32 noundef %r4, i32 noundef %r5, i32 noundef %r6, [4 x i32] %0)
#[unsafe(no_mangle)]
pub extern "C" fn fits_last(r3: i32, r4: i32, r5: i32, r6: i32, r7_r10: Cd) {}

// The complex value does not fit, so the remaining GPRs are filled with padding.

// CHECK: define{{.*}} void @pad_three(i32 noundef %r3, i32 noundef %r4, i32 noundef %r5, i32 noundef %r6, i32 noundef %r7, i32 %0, i32 %1, i32 %2, [4 x i32] %3, i32 noundef %stack16)
#[unsafe(no_mangle)]
pub extern "C" fn pad_three(r3: i32, r4: i32, r5: i32, r6: i32, r7: i32, stack0: Cd, stack16: i32) {
}

// CHECK: define{{.*}} void @pad_second(i32 noundef %r3, [4 x i32] %0, i32 noundef %r8, i32 noundef %r9, i32 %1, [4 x i32] %2, i32 noundef %stack16)
#[unsafe(no_mangle)]
pub extern "C" fn pad_second(r3: i32, r4_r7: Cd, r8: i32, r9: i32, stack0: Cd, stack16: i32) {}

// Arguments are passed in r3..=r10, and after that on the stack, so no padding is needed here.

// CHECK: define{{.*}} void @no_gprs_left(i32 noundef %r3, i32 noundef %r4, i32 noundef %r5, i32 noundef %r6, i32 noundef %r7, i32 noundef %r8, i32 noundef %r9, i32 noundef %r10, i32 noundef %stack0, [4 x i32] %0)
#[unsafe(no_mangle)]
pub extern "C" fn no_gprs_left(
    r3: i32,
    r4: i32,
    r5: i32,
    r6: i32,
    r7: i32,
    r8: i32,
    r9: i32,
    r10: i32,
    stack0: i32,
    stack4: Cd,
) {
}

// An 8-byte complex value is cast to i64 so that it uses an even-odd GPR pair.
// Strangely, larger complex values do not need to be aligned.

// CHECK: define{{.*}} void @complex_cf_aligned(i32 noundef %r3, i64 %0, i32 %1, i32 %2, i32 %3, i32 %4, [8 x i32] %5)
#[unsafe(no_mangle)]
pub extern "C" fn complex_cf_aligned(r3: i32, r5_r6: Cf, x: Cld) {}

// CHECK: define{{.*}} void @complex_cd_unaligned(i32 noundef %r3, [4 x i32] %0, i32 %1, i32 %2, i32 %3, [8 x i32] %4)
#[unsafe(no_mangle)]
pub extern "C" fn complex_cd_unaligned(r3: i32, r4_r7: Cd, x: Cld) {}

// A `Cf` that does not fit in r10 is passed on the stack.

// CHECK: define{{.*}} void @complex_cf_spill(i32 noundef %r3, i32 noundef %r4, i32 noundef %r5, i32 noundef %r6, i32 noundef %r7, i32 noundef %r8, i32 noundef %r9, i64 %0, [8 x i32] %1)
#[unsafe(no_mangle)]
pub extern "C" fn complex_cf_spill(
    r3: i32,
    r4: i32,
    r5: i32,
    r6: i32,
    r7: i32,
    r8: i32,
    r9: i32,
    stack0: Cf,
    x: Cld,
) {
}

// An `f32` is passed in an FPR, so the `Cf` uses r3..=r4. Then r5..=r10 are padding, with the Cld
// passed via the stack.

// CHECK: define{{.*}} void @complex_cf_after_float(float noundef %f1, i64 %0, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, [8 x i32] %7)
#[unsafe(no_mangle)]
pub extern "C" fn complex_cf_after_float(f1: f32, r3_r4: Cf, x: Cld) {}

// Check that the sret pointer uses a GPR.

// CHECK: define{{.*}} void @sret(ptr {{.*}}sret([32 x i8]){{.*}}, i32 %0, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, [8 x i32] %7)
#[unsafe(no_mangle)]
pub extern "C" fn sret(x: Cld) -> Big {
    loop {}
}

// A `repr(C)` ZST is passed by reference, and the pointer uses a GPR.

// CHECK: define{{.*}} void @zst(ptr {{.*}}%_1, i32 %0, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, [8 x i32] %7)
#[unsafe(no_mangle)]
pub extern "C" fn zst(r3: Empty, x: Cld) {}

// CHECK: define{{.*}} void @zst_no_gprs_left(i32 noundef %r3, i32 noundef %r4, i32 noundef %r5, i32 noundef %r6, i32 noundef %r7, i32 noundef %r8, i32 noundef %r9, i32 noundef %r10, ptr {{.*}}%_9, [4 x i32] %0)
#[unsafe(no_mangle)]
pub extern "C" fn zst_no_gprs_left(
    r3: i32,
    r4: i32,
    r5: i32,
    r6: i32,
    r7: i32,
    r8: i32,
    r9: i32,
    r10: i32,
    stack0: Empty,
    stack4: Cd,
) {
}

// Integers wider than a GPR (i.e. 32 bits) are split by LLVM, which aligns them to an even-odd GPR pair.

// CHECK: define{{.*}} void @i64_aligned(i32 noundef %r3, i64 noundef %r5_r6, i32 %0, i32 %1, i32 %2, i32 %3, [8 x i32] %4)
#[unsafe(no_mangle)]
pub extern "C" fn i64_aligned(r3: i32, r5_r6: i64, x: Cld) {}

// CHECK: define{{.*}} void @i128_aligned(i32 noundef %r3, i128 noundef %r5_r8, i32 %0, i32 %1, [8 x i32] %2)
#[unsafe(no_mangle)]
pub extern "C" fn i128_aligned(r3: i32, r5_r8: i128, x: Cld) {}

// CHECK: define{{.*}} void @i64_spill(i32 noundef %r3, i32 noundef %r4, i32 noundef %r5, i32 noundef %r6, i32 noundef %r7, i32 noundef %r8, i32 noundef %r9, i64 noundef %stack0, [8 x i32] %0)
#[unsafe(no_mangle)]
pub extern "C" fn i64_spill(
    r3: i32,
    r4: i32,
    r5: i32,
    r6: i32,
    r7: i32,
    r8: i32,
    r9: i32,
    stack0: i64,
    x: Cld,
) {
}

// Lock in the current LLVM behavior of splitting up `i128` between registers and stack.
//
// Note that neither the ABI specification nor C compilers support `i128` at the moment,
// so the behavior of `i128` may change in the future.

// CHECK: define{{.*}} void @i128_split(i32 noundef %r3, i32 noundef %r4, i32 noundef %r5, i32 noundef %r6, i32 noundef %r7, i128 noundef %r9_r10_stack0, [8 x i32] %0)
#[unsafe(no_mangle)]
pub extern "C" fn i128_split(
    r3: i32,
    r4: i32,
    r5: i32,
    r6: i32,
    r7: i32,
    r9_r10_stack0: i128,
    x: Cld,
) {
}

// Only `f32`, `f64` and `ppcf128` are passed in FPRs, `f16` uses GPRs (although `f16` might be
// changed to use FPRs in the future), `f128` is only valid with +vsx.

// CHECK: define{{.*}} void @f16_gpr(half noundef %r3, i32 %0, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, [8 x i32] %7)
#[unsafe(no_mangle)]
pub extern "C" fn f16_gpr(r3: f16, x: Cld) {}

// CHECK: define{{.*}} void @f32_fpr(float noundef %f1, [8 x i32] %0)
#[unsafe(no_mangle)]
pub extern "C" fn f32_fpr(f1: f32, x: Cld) {}

// CHECK: define{{.*}} void @f64_fpr(double noundef %f1, [8 x i32] %0)
#[unsafe(no_mangle)]
pub extern "C" fn f64_fpr(f1: f64, x: Cld) {}

// CHECK: define{{.*}} void @ppcf128_fpr(ppc_fp128 noundef %f1_f2, [8 x i32] %0)
#[unsafe(no_mangle)]
pub extern "C" fn ppcf128_fpr(f1_f2: ppcf128, x: Cld) {}

// Legal vector types are passed in vector registers, they never use GPRs.

#[cfg(target_feature = "altivec")]
mod altivec {
    use super::*;

    // ALTIVEC: define{{.*}} void @vec_i8x16(<16 x i8> %v2, [8 x i32] %0)
    #[unsafe(no_mangle)]
    pub extern "C" fn vec_i8x16(v2: i8x16, x: Cld) {}

    // ALTIVEC: define{{.*}} void @vec_i16x8(<8 x i16> %v2, [8 x i32] %0)
    #[unsafe(no_mangle)]
    pub extern "C" fn vec_i16x8(v2: i16x8, x: Cld) {}

    // ALTIVEC: define{{.*}} void @vec_i32x4(<4 x i32> %v2, [8 x i32] %0)
    #[unsafe(no_mangle)]
    pub extern "C" fn vec_i32x4(v2: i32x4, x: Cld) {}

    // ALTIVEC: define{{.*}} void @vec_f64x2(<2 x double> %f1_f2, [8 x i32] %0)
    #[unsafe(no_mangle)]
    pub extern "C" fn vec_f64x2(f1_f2: f64x2, x: Cld) {}
}

#[cfg(target_feature = "vsx")]
mod vsx {
    use super::*;

    // VSX: define{{.*}} void @vec_i64x2(<2 x i64> %v2, [8 x i32] %0)
    #[unsafe(no_mangle)]
    pub extern "C" fn vec_i64x2(v2: i64x2, x: Cld) {}

    // VSX: define{{.*}} void @plain_f128(fp128 noundef %v2, [8 x i32] %0)
    #[unsafe(no_mangle)]
    pub extern "C" fn plain_f128(v2: f128, x: Cld) {}

    // VSX: define{{.*}} void @vec_f128x1(<1 x fp128> %v2, [8 x i32] %0)
    #[unsafe(no_mangle)]
    pub extern "C" fn vec_f128x1(v2: Simd<f128, 1>, x: Cld) {}
}
