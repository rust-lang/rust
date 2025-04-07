//@ add-core-stubs
//@ revisions: x86 x86-avx2 x86-avx512 aarch64
//@ [x86] compile-flags: --target=x86_64-unknown-linux-gnu -C llvm-args=-x86-asm-syntax=intel
//@ [x86] needs-llvm-components: x86
//@ [x86-avx2] compile-flags: --target=x86_64-unknown-linux-gnu -C llvm-args=-x86-asm-syntax=intel
//@ [x86-avx2] compile-flags: -C target-feature=+avx2
//@ [x86-avx2] needs-llvm-components: x86
//@ [x86-avx512] compile-flags: --target=x86_64-unknown-linux-gnu -C llvm-args=-x86-asm-syntax=intel
//@ [x86-avx512] compile-flags: -C target-feature=+avx512f,+avx512vl,+avx512bw,+avx512dq
//@ [x86-avx512] needs-llvm-components: x86
//@ [aarch64] compile-flags: --target=aarch64-unknown-linux-gnu
//@ [aarch64] needs-llvm-components: aarch64
//@ assembly-output: emit-asm
//@ compile-flags: --crate-type=lib -Copt-level=3 -C panic=abort

#![feature(no_core, lang_items, repr_simd, intrinsics)]
#![no_core]
#![allow(non_camel_case_types)]

extern crate minicore;
use minicore::*;

#[repr(simd)]
pub struct m8x16([i8; 16]);

#[repr(simd)]
pub struct m8x64([i8; 64]);

#[repr(simd)]
pub struct m32x4([i32; 4]);

#[repr(simd)]
pub struct m64x2([i64; 2]);

#[repr(simd)]
pub struct m64x4([i64; 4]);

#[rustc_intrinsic]
unsafe fn simd_bitmask<V, B>(mask: V) -> B;

// CHECK-LABEL: bitmask_m8x16
#[no_mangle]
pub unsafe extern "C" fn bitmask_m8x16(mask: m8x16) -> u16 {
    // The simd_bitmask intrinsic already uses the most significant bit, so no shift is necessary.
    // Note that x86 has no byte shift, llvm uses a word shift to move the least significant bit
    // of each byte into the right position.
    //
    // x86-NOT: psllw
    // x86: movmskb eax, xmm0
    //
    // x86-avx2-NOT: vpsllw
    // x86-avx2: vpmovmskb eax, xmm0
    //
    // x86-avx512-NOT: vpsllw xmm0
    // x86-avx512: vpmovmskb eax, xmm0
    //
    // aarch64: adrp
    // aarch64-NEXT: cmlt
    // aarch64-NEXT: ldr
    // aarch64-NEXT: and
    // aarch64-NEXT: ext
    // aarch64-NEXT: zip1
    // aarch64-NEXT: addv
    // aarch64-NEXT: fmov
    simd_bitmask(mask)
}

// CHECK-LABEL: bitmask_m8x64
#[no_mangle]
pub unsafe extern "C" fn bitmask_m8x64(mask: m8x64) -> u64 {
    // The simd_bitmask intrinsic already uses the most significant bit, so no shift is necessary.
    // Note that x86 has no byte shift, llvm uses a word shift to move the least significant bit
    // of each byte into the right position.
    //
    // The parameter is a 512 bit vector which in the C abi is only valid for avx512 targets.
    //
    // x86-avx512-NOT: vpsllw
    // x86-avx512: vpmovb2m k0, zmm0
    // x86-avx512: kmovq rax, k0
    simd_bitmask(mask)
}

// CHECK-LABEL: bitmask_m32x4
#[no_mangle]
pub unsafe extern "C" fn bitmask_m32x4(mask: m32x4) -> u8 {
    // The simd_bitmask intrinsic already uses the most significant bit, so no shift is necessary.
    //
    // x86-NOT: psllq
    // x86: movmskps eax, xmm0
    //
    // x86-avx2-NOT: vpsllq
    // x86-avx2: vmovmskps eax, xmm0
    //
    // x86-avx512-NOT: vpsllq
    // x86-avx512: vmovmskps eax, xmm0
    //
    // aarch64: adrp
    // aarch64-NEXT: cmlt
    // aarch64-NEXT: ldr
    // aarch64-NEXT: and
    // aarch64-NEXT: addv
    // aarch64-NEXT: fmov
    // aarch64-NEXT: and
    simd_bitmask(mask)
}

// CHECK-LABEL: bitmask_m64x2
#[no_mangle]
pub unsafe extern "C" fn bitmask_m64x2(mask: m64x2) -> u8 {
    // The simd_bitmask intrinsic already uses the most significant bit, so no shift is necessary.
    //
    // x86-NOT: psllq
    // x86: movmskpd eax, xmm0
    //
    // x86-avx2-NOT: vpsllq
    // x86-avx2: vmovmskpd eax, xmm0
    //
    // x86-avx512-NOT: vpsllq
    // x86-avx512: vmovmskpd eax, xmm0
    //
    // aarch64: adrp
    // aarch64-NEXT: cmlt
    // aarch64-NEXT: ldr
    // aarch64-NEXT: and
    // aarch64-NEXT: addp
    // aarch64-NEXT: fmov
    // aarch64-NEXT: and
    simd_bitmask(mask)
}

// CHECK-LABEL: bitmask_m64x4
#[no_mangle]
pub unsafe extern "C" fn bitmask_m64x4(mask: m64x4) -> u8 {
    // The simd_bitmask intrinsic already uses the most significant bit, so no shift is necessary.
    //
    // The parameter is a 256 bit vector which in the C abi is only valid for avx/avx512 targets.
    //
    // x86-avx2-NOT: vpsllq
    // x86-avx2: vmovmskpd eax, ymm0
    //
    // x86-avx512-NOT: vpsllq
    // x86-avx512: vmovmskpd eax, ymm0
    simd_bitmask(mask)
}
