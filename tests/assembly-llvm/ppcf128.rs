//@ add-minicore
//@ assembly-output: emit-asm
//
//@ revisions: POWERPC POWERPC64LE POWERPC64 AIX
//@ [POWERPC] compile-flags: --target powerpc-unknown-linux-gnu
//@ [POWERPC64LE] compile-flags: --target powerpc64le-unknown-linux-gnu
//@ [POWERPC64] compile-flags: --target powerpc64-unknown-linux-gnu
//@ [AIX] compile-flags: --target powerpc64-ibm-aix
//@ compile-flags: -Copt-level=3 --crate-type=lib
//@ needs-llvm-components: powerpc

#![feature(no_core)]
#![no_std]
#![no_core]

extern crate minicore;
#[cfg(target_arch = "powerpc")]
use minicore::arch::powerpc::ppcf128;
#[cfg(target_arch = "powerpc64")]
use minicore::arch::powerpc64::ppcf128;
use minicore::*;

// CHECK-LABEL: identity
//
// POWERPC:      .Lfunc_begin{{[0-9]+}}:
// POWERPC-NEXT:  .cfi_startproc
// POWERPC-NEXT:  stwu 1, -48(1)
// POWERPC-NEXT:  .cfi_def_cfa_offset 48
// POWERPC-NEXT:  stfd 1, 24(1)
// POWERPC-NEXT:  lwz 3, 28(1)
// POWERPC-NEXT:  stfd 2, 16(1)
// POWERPC-NEXT:  stw 3, 44(1)
// POWERPC-NEXT:  lwz 3, 24(1)
// POWERPC-NEXT:  stw 3, 40(1)
// POWERPC-NEXT:  lwz 3, 20(1)
// POWERPC-NEXT:  lfd 1, 40(1)
// POWERPC-NEXT:  stw 3, 36(1)
// POWERPC-NEXT:  lwz 3, 16(1)
// POWERPC-NEXT:  stw 3, 32(1)
// POWERPC-NEXT:  lfd 2, 32(1)
// POWERPC-NEXT:  addi 1, 1, 48
// POWERPC-NEXT:  blr
//
// POWERPC64:      .Lfunc_begin{{[0-9]+}}:
// POWERPC64-NEXT:  .cfi_startproc
// POWERPC64-NEXT:  blr
//
// POWERPC64LE:      .Lfunc_begin{{[0-9]+}}:
// POWERPC64LE-NEXT:  .cfi_startproc
// POWERPC64LE-NEXT:  blr
//
// AIX:      .csect .identity[PR],5
// AIX-NEXT: blr
#[unsafe(no_mangle)]
pub extern "C" fn identity(x: ppcf128) -> ppcf128 {
    x
}

/// On elfv1 and aix single-float structs are passed as scalar arguments.
#[repr(C)]
struct Hfa1 {
    a: ppcf128,
}

/// On elfv2 homogenous aggregates of up to 4 elements are passed as scalars.
#[repr(C)]
struct Hfa2 {
    a: ppcf128,
    b: ppcf128,
}

#[repr(C)]
struct Hfa4 {
    a: ppcf128,
    b: ppcf128,
    c: ppcf128,
    d: ppcf128,
}

// CHECK-LABEL: hfa1
//
// POWERPC:      .Lfunc_begin{{[0-9]+}}:
// POWERPC-NEXT:  .cfi_startproc
// POWERPC-NEXT:  stwu 1, -32(1)
// POWERPC-NEXT:  .cfi_def_cfa_offset 32
// POWERPC-NEXT:  lwz 4, 4(3)
// POWERPC-NEXT:  stw 4, 20(1)
// POWERPC-NEXT:  lwz 4, 0(3)
// POWERPC-NEXT:  stw 4, 16(1)
// POWERPC-NEXT:  lwz 4, 12(3)
// POWERPC-NEXT:  lfd 1, 16(1)
// POWERPC-NEXT:  stw 4, 28(1)
// POWERPC-NEXT:  lwz 3, 8(3)
// POWERPC-NEXT:  stw 3, 24(1)
// POWERPC-NEXT:  lfd 2, 24(1)
// POWERPC-NEXT:  addi 1, 1, 32
// POWERPC-NEXT:  blr
//
// POWERPC64:      .Lfunc_begin{{[0-9]+}}:
// POWERPC64-NEXT:  .cfi_startproc
// POWERPC64-NEXT:  blr
//
// POWERPC64LE:      .Lfunc_begin{{[0-9]+}}:
// POWERPC64LE-NEXT:  .cfi_startproc
// POWERPC64LE-NEXT:  blr
//
// AIX:      .csect .hfa1[PR],5
// AIX-NEXT: std 3, -16(1)
// AIX-NEXT: std 4, -8(1)
// AIX-NEXT: lfd 1, -16(1)
// AIX-NEXT: lfd 2, -8(1)
// AIX-NEXT: std 3, 48(1)
// AIX-NEXT: std 4, 56(1)
// AIX-NEXT: blr
#[unsafe(no_mangle)]
pub extern "C" fn hfa1(hfa: Hfa1) -> ppcf128 {
    hfa.a
}

// CHECK-LABEL: hfa2
//
// POWERPC:      .Lfunc_begin{{[0-9]+}}:
// POWERPC-NEXT:  .cfi_startproc
// POWERPC-NEXT:  stwu 1, -32(1)
// POWERPC-NEXT:  .cfi_def_cfa_offset 32
// POWERPC-NEXT:  lwz 4, 20(3)
// POWERPC-NEXT:  stw 4, 20(1)
// POWERPC-NEXT:  lwz 4, 16(3)
// POWERPC-NEXT:  stw 4, 16(1)
// POWERPC-NEXT:  lwz 4, 28(3)
// POWERPC-NEXT:  lfd 1, 16(1)
// POWERPC-NEXT:  stw 4, 28(1)
// POWERPC-NEXT:  lwz 3, 24(3)
// POWERPC-NEXT:  stw 3, 24(1)
// POWERPC-NEXT:  lfd 2, 24(1)
// POWERPC-NEXT:  addi 1, 1, 32
// POWERPC-NEXT:  blr
//
// POWERPC64:      .Lfunc_begin{{[0-9]+}}:
// POWERPC64-NEXT:  .cfi_startproc
// POWERPC64-NEXT:  std 5, -16(1)
// POWERPC64-NEXT:  std 6, -8(1)
// POWERPC64-NEXT:  lfd 1, -16(1)
// POWERPC64-NEXT:  lfd 2, -8(1)
// POWERPC64-NEXT:  blr
//
// POWERPC64LE:      .Lfunc_begin{{[0-9]+}}:
// POWERPC64LE-NEXT:  .cfi_startproc
// POWERPC64LE-NEXT:  fmr 2, 4
// POWERPC64LE-NEXT:  fmr 1, 3
// POWERPC64LE-NEXT:  blr
//
// AIX:      .csect .hfa2[PR],5
// AIX-NEXT: std 5, -16(1)
// AIX-NEXT: std 6, -8(1)
// AIX-NEXT: lfd 1, -16(1)
// AIX-NEXT: lfd 2, -8(1)
// AIX-NEXT: std 6, 72(1)
// AIX-NEXT: std 5, 64(1)
// AIX-NEXT: std 3, 48(1)
// AIX-NEXT: std 4, 56(1)
// AIX-NEXT: blr
#[unsafe(no_mangle)]
pub extern "C" fn hfa2(value: Hfa2) -> ppcf128 {
    value.b
}

// CHECK-LABEL: hfa4
//
// POWERPC:      .Lfunc_begin{{[0-9]+}}:
// POWERPC-NEXT:  .cfi_startproc
// POWERPC-NEXT:  stwu 1, -32(1)
// POWERPC-NEXT:  .cfi_def_cfa_offset 32
// POWERPC-NEXT:  lwz 4, 52(3)
// POWERPC-NEXT:  stw 4, 20(1)
// POWERPC-NEXT:  lwz 4, 48(3)
// POWERPC-NEXT:  stw 4, 16(1)
// POWERPC-NEXT:  lwz 4, 60(3)
// POWERPC-NEXT:  lfd 1, 16(1)
// POWERPC-NEXT:  stw 4, 28(1)
// POWERPC-NEXT:  lwz 3, 56(3)
// POWERPC-NEXT:  stw 3, 24(1)
// POWERPC-NEXT:  lfd 2, 24(1)
// POWERPC-NEXT:  addi 1, 1, 32
// POWERPC-NEXT:  blr
//
// POWERPC64:      .Lfunc_begin{{[0-9]+}}:
// POWERPC64-NEXT:  .cfi_startproc
// POWERPC64-NEXT:  std 9, -16(1)
// POWERPC64-NEXT:  std 10, -8(1)
// POWERPC64-NEXT:  lfd 1, -16(1)
// POWERPC64-NEXT:  lfd 2, -8(1)
// POWERPC64-NEXT:  blr
//
// POWERPC64LE:      .Lfunc_begin{{[0-9]+}}:
// POWERPC64LE-NEXT:  .cfi_startproc
// POWERPC64LE-NEXT:  fmr 2, 8
// POWERPC64LE-NEXT:  fmr 1, 7
// POWERPC64LE-NEXT:  blr
//
// AIX:      .csect .hfa4[PR],5
// AIX-NEXT: std 9, -16(1)
// AIX-NEXT: std 10, -8(1)
// AIX-NEXT: lfd 1, -16(1)
// AIX-NEXT: lfd 2, -8(1)
// AIX-NEXT: std 10, 104(1)
// AIX-NEXT: std 9, 96(1)
// AIX-NEXT: std 3, 48(1)
// AIX-NEXT: std 4, 56(1)
// AIX-NEXT: std 5, 64(1)
// AIX-NEXT: std 6, 72(1)
// AIX-NEXT: std 7, 80(1)
// AIX-NEXT: std 8, 88(1)
// AIX-NEXT: blr
#[unsafe(no_mangle)]
pub extern "C" fn hfa4(value: Hfa4) -> ppcf128 {
    value.d
}

#[repr(C)]
struct NonHfa5 {
    a: ppcf128,
    b: ppcf128,
    c: ppcf128,
    d: ppcf128,
    e: ppcf128,
}

// CHECK-LABEL: non_hfa5
//
// POWERPC:      .Lfunc_begin{{[0-9]+}}:
// POWERPC-NEXT:  .cfi_startproc
// POWERPC-NEXT:  stwu 1, -32(1)
// POWERPC-NEXT:  .cfi_def_cfa_offset 32
// POWERPC-NEXT:  lwz 4, 68(3)
// POWERPC-NEXT:  stw 4, 20(1)
// POWERPC-NEXT:  lwz 4, 64(3)
// POWERPC-NEXT:  stw 4, 16(1)
// POWERPC-NEXT:  lwz 4, 76(3)
// POWERPC-NEXT:  lfd 1, 16(1)
// POWERPC-NEXT:  stw 4, 28(1)
// POWERPC-NEXT:  lwz 3, 72(3)
// POWERPC-NEXT:  stw 3, 24(1)
// POWERPC-NEXT:  lfd 2, 24(1)
// POWERPC-NEXT:  addi 1, 1, 32
// POWERPC-NEXT:  blr
//
// POWERPC64:      .Lfunc_begin{{[0-9]+}}:
// POWERPC64-NEXT:  .cfi_startproc
// POWERPC64-NEXT:  lfd 1, 112(1)
// POWERPC64-NEXT:  lfd 2, 120(1)
// POWERPC64-NEXT:  blr
//
// POWERPC64LE:      .Lfunc_begin{{[0-9]+}}:
// POWERPC64LE-NEXT:  .cfi_startproc
// POWERPC64LE-NEXT:  lfd 1, 96(1)
// POWERPC64LE-NEXT:  lfd 2, 104(1)
// POWERPC64LE-NEXT:  blr
//
// AIX:      .csect .non_hfa5[PR],5
// AIX-NEXT: lfd 1, 112(1)
// AIX-NEXT: lfd 2, 120(1)
// AIX-NEXT: std 3, 48(1)
// AIX-NEXT: std 4, 56(1)
// AIX-NEXT: std 5, 64(1)
// AIX-NEXT: std 6, 72(1)
// AIX-NEXT: std 7, 80(1)
// AIX-NEXT: std 8, 88(1)
// AIX-NEXT: std 9, 96(1)
// AIX-NEXT: std 10, 104(1)
// AIX-NEXT: blr
#[unsafe(no_mangle)]
pub extern "C" fn non_hfa5(value: NonHfa5) -> ppcf128 {
    value.e
}
