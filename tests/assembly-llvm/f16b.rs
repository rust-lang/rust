//@ add-minicore
//@ assembly-output: emit-asm
//
//@ revisions: AARCH64_LINUX AARCH64_BE AARCH64_DARWIN AARCH64_MSVC ARM64EC_MSVC X64_LINUX X64_WINDOWS_GNU X64_WINDOWS_MSVC RISCV64 LOONGARCH64
//@ [AARCH64_LINUX] compile-flags: -Copt-level=3 --target aarch64-unknown-linux-gnu
//@ [AARCH64_LINUX] needs-llvm-components: aarch64
//@ [AARCH64_LINUX] filecheck-flags: --check-prefixes AARCH64,AARCH64-NOTAPPLE
//@ [AARCH64_BE] compile-flags: -Copt-level=3 --target aarch64_be-unknown-linux-gnu
//@ [AARCH64_BE] needs-llvm-components: aarch64
//@ [AARCH64_BE] filecheck-flags: --check-prefixes AARCH64,AARCH64-NOTAPPLE
//@ [AARCH64_DARWIN] compile-flags: -Copt-level=3 --target aarch64-apple-darwin
//@ [AARCH64_DARWIN] needs-llvm-components: aarch64
//@ [AARCH64_DARWIN] filecheck-flags: --check-prefixes AARCH64,AARCH64-APPLE
//@ [AARCH64_MSVC] compile-flags: -Copt-level=3 --target aarch64-pc-windows-msvc
//@ [AARCH64_MSVC] needs-llvm-components: aarch64
//@ [AARCH64_MSVC] filecheck-flags: --check-prefixes AARCH64,AARCH64-NOTAPPLE
//@ [ARM64EC_MSVC] compile-flags: -Copt-level=3 --target arm64ec-pc-windows-msvc
//@ [ARM64EC_MSVC] needs-llvm-components: aarch64
//@ [ARM64EC_MSVC] min-llvm-version: 23
//@ [ARM64EC_MSVC] filecheck-flags: --check-prefixes AARCH64,AARCH64-NOTAPPLE
//@ [X64_LINUX] compile-flags: -Copt-level=3 -Cllvm-args=-x86-asm-syntax=intel --target x86_64-unknown-linux-gnu
//@ [X64_LINUX] needs-llvm-components: x86
//@ [X64_LINUX] filecheck-flags: --check-prefixes X64,X64-LINUX
//@ [X64_WINDOWS_GNU] compile-flags: -Copt-level=3 -Cllvm-args=-x86-asm-syntax=intel --target x86_64-pc-windows-gnu
//@ [X64_WINDOWS_GNU] needs-llvm-components: x86
//@ [X64_WINDOWS_GNU] filecheck-flags: --check-prefixes X64,X64-WINDOWS
//@ [X64_WINDOWS_MSVC] compile-flags: -Copt-level=3 -Cllvm-args=-x86-asm-syntax=intel --target x86_64-pc-windows-msvc
//@ [X64_WINDOWS_MSVC] needs-llvm-components: x86
//@ [X64_WINDOWS_MSVC] filecheck-flags: --check-prefixes X64,X64-WINDOWS
//@ [RISCV64] compile-flags: -Copt-level=3 --target riscv64gc-unknown-linux-gnu
//@ [RISCV64] needs-llvm-components: riscv
//@ [LOONGARCH64] compile-flags: -Copt-level=3 --target loongarch64-unknown-linux-gnu
//@ [LOONGARCH64] needs-llvm-components: loongarch

#![feature(f16b, no_core)]
#![no_core]
#![crate_type = "lib"]
#![allow(improper_ctypes_definitions)]

// Check that the assembly that rustc generates matches what clang emits.

extern crate minicore;

use minicore::From;
use minicore::num::f16b;

// CHECK-LABEL: {{^"?[#_]?identity_f16b"?:}}
// AARCH64: ret
// X64: ret
// RISCV64: fmv.x.w a0, fa0
// RISCV64-NEXT: lui a1, 1048560
// RISCV64-NEXT: or a0, a0, a1
// RISCV64-NEXT: fmv.w.x fa0, a0
// RISCV64-NEXT: ret
// LOONGARCH64: movfr2gr.s $a0, $fa0
// LOONGARCH64-NEXT: lu12i.w $a1, -16
// LOONGARCH64-NEXT: or $a0, $a0, $a1
// LOONGARCH64-NEXT: movgr2fr.w $fa0, $a0
// LOONGARCH64-NEXT: ret
#[unsafe(no_mangle)]
pub extern "C" fn identity_f16b(value: f16b) -> f16b {
    value
}

// CHECK-LABEL: {{^"?[#_]?f16b_to_bits"?:}}
// AARCH64-NOTAPPLE: fmov w0, s0
// AARCH64-APPLE: fmov w8, s0
// AARCH64-APPLE-NEXT: and w0, w8, #0xffff
// AARCH64-NEXT: ret
// X64: pextrw eax, xmm0, 0
// X64-NEXT: ret
// RISCV64: fmv.x.w a0, fa0
// RISCV64-NEXT: slli a0, a0, 48
// RISCV64-NEXT: srli a0, a0, 48
// RISCV64-NEXT: ret
// LOONGARCH64: movfr2gr.s $a0, $fa0
// LOONGARCH64-NEXT: bstrpick.d $a0, $a0, 15, 0
// LOONGARCH64-NEXT: ret
#[unsafe(no_mangle)]
pub extern "C" fn f16b_to_bits(value: f16b) -> u16 {
    value.to_bits()
}

// CHECK-LABEL: {{^"?[#_]?f16b_from_bits"?:}}
// AARCH64: fmov s0, w0
// AARCH64-NEXT: ret
// X64-LINUX: pinsrw xmm0, edi, 0
// X64-LINUX-NEXT: ret
// X64-WINDOWS: pinsrw xmm0, ecx, 0
// X64-WINDOWS-NEXT: ret
// RISCV64: lui a1, 1048560
// RISCV64-NEXT: or a0, a0, a1
// RISCV64-NEXT: fmv.w.x fa0, a0
// RISCV64-NEXT: ret
// LOONGARCH64: lu12i.w $a1, -16
// LOONGARCH64-NEXT: or $a0, $a0, $a1
// LOONGARCH64-NEXT: movgr2fr.w $fa0, $a0
// LOONGARCH64-NEXT: ret
#[unsafe(no_mangle)]
pub extern "C" fn f16b_from_bits(bits: u16) -> f16b {
    f16b::from_bits(bits)
}

// CHECK-LABEL: {{^"?[#_]?widen_f16b"?:}}
// AARCH64: fmov w8, s0
// AARCH64-NEXT: lsl w8, w8, #16
// AARCH64-NEXT: fmov s0, w8
// AARCH64-NEXT: ret
// X64: pextrw eax, xmm0, 0
// X64-NEXT: shl eax, 16
// X64-NEXT: movd xmm0, eax
// X64-NEXT: ret
// RISCV64: fmv.x.w a0, fa0
// RISCV64-NEXT: slli a0, a0, 16
// RISCV64-NEXT: fmv.w.x fa0, a0
// RISCV64-NEXT: ret
// LOONGARCH64: movfr2gr.s $a0, $fa0
// LOONGARCH64-NEXT: slli.d $a0, $a0, 16
// LOONGARCH64-NEXT: movgr2fr.w $fa0, $a0
// LOONGARCH64-NEXT: ret
#[unsafe(no_mangle)]
pub extern "C" fn widen_f16b(value: f16b) -> f32 {
    f32::from(value)
}
