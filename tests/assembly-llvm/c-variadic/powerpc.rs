//@ add-minicore
//@ assembly-output: emit-asm
//
//@ revisions: POWERPC POWERPC64 POWERPC64LE AIX
//@ [POWERPC] compile-flags: -Copt-level=3 --target powerpc-unknown-linux-gnu
//@ [POWERPC] needs-llvm-components: powerpc
//@ [POWERPC64] compile-flags: -Copt-level=3 --target powerpc64-unknown-linux-gnu
//@ [POWERPC64] needs-llvm-components: powerpc
//@ [POWERPC64LE] compile-flags: -Copt-level=3 --target powerpc64le-unknown-linux-gnu
//@ [POWERPC64LE] needs-llvm-components: powerpc
//@ [AIX] compile-flags: -Copt-level=3 --target powerpc64-ibm-aix
//@ [AIX] needs-llvm-components: powerpc
#![feature(no_core, lang_items, intrinsics, rustc_attrs, f128)]
#![no_core]
#![crate_type = "lib"]

// Check that the assembly that rustc generates matches what clang emits.

extern crate minicore;
#[cfg(target_arch = "powerpc")]
use minicore::arch::powerpc::ppcf128;
#[cfg(target_arch = "powerpc64")]
use minicore::arch::powerpc64::ppcf128;
use minicore::ffi::VaList;
use minicore::*;

#[unsafe(no_mangle)]
unsafe extern "C" fn read_f64(ap: &mut VaList<'_>) -> f64 {
    // CHECK-LABEL: read_f64
    //
    // POWERPC: lbz 5, 1(3)
    // POWERPC-NEXT: cmplwi 5, 7
    // POWERPC-NEXT: bgt 0, .LBB0_2
    // POWERPC-NEXT: lwz 4, 8(3)
    // POWERPC-NEXT: rlwinm 6, 5, 3, 24, 28
    // POWERPC-NEXT: addi 5, 5, 1
    // POWERPC-NEXT: add 4, 4, 6
    // POWERPC-NEXT: addi 4, 4, 32
    // POWERPC-NEXT: lfd 1, 0(4)
    // POWERPC-NEXT: stb 5, 1(3)
    // POWERPC-NEXT: blr
    //
    // POWERPC64: ld 4, 0(3)
    // POWERPC64-NEXT: lfd 1, 0(4)
    // POWERPC64-NEXT: addi 4, 4, 8
    // POWERPC64-NEXT: std 4, 0(3)
    // POWERPC64-NEXT: blr
    //
    // POWERPC64LE: ld 4, 0(3)
    // POWERPC64LE-NEXT: lfd 1, 0(4)
    // POWERPC64LE-NEXT: addi 5, 4, 8
    // POWERPC64LE-NEXT: std 5, 0(3)
    // POWERPC64LE-NEXT: blr
    //
    // AIX: ld 4, 0(3)
    // AIX-NEXT: lfd 1, 0(4)
    // AIX-NEXT: addi 5, 4, 8
    // AIX-NEXT: std 5, 0(3)
    // AIX-NEXT: blr
    ap.next_arg()
}

#[unsafe(no_mangle)]
unsafe extern "C" fn read_i32(ap: &mut VaList<'_>) -> i32 {
    // CHECK-LABEL: read_i32
    //
    // POWERPC: lbz 5, 0(3)
    // POWERPC-NEXT: mr 4, 3
    // POWERPC-NEXT: cmplwi 5, 7
    // POWERPC-NEXT: bgt 0, .LBB1_2
    // POWERPC-NEXT: lwz 3, 8(4)
    // POWERPC-NEXT: rlwinm 6, 5, 2, 24, 29
    // POWERPC-NEXT: addi 5, 5, 1
    // POWERPC-NEXT: add 3, 3, 6
    // POWERPC-NEXT: lwz 3, 0(3)
    // POWERPC-NEXT: stb 5, 0(4)
    // POWERPC-NEXT: blr
    //
    // POWERPC64: ld 5, 0(3)
    // POWERPC64-NEXT: mr 4, 3
    // POWERPC64-NEXT: lwa 3, 4(5)
    // POWERPC64-NEXT: addi 5, 5, 8
    // POWERPC64-NEXT: std 5, 0(4)
    // POWERPC64-NEXT: blr
    //
    // POWERPC64LE: ld 4, 0(3)
    // POWERPC64LE-NEXT: addi 5, 4, 8
    // POWERPC64LE-NEXT: std 5, 0(3)
    // POWERPC64LE-NEXT: lwa 3, 0(4)
    // POWERPC64LE-NEXT: blr
    //
    // AIX: ld 4, 0(3)
    // AIX-NEXT: addi 5, 4, 8
    // AIX-NEXT: std 5, 0(3)
    // AIX-NEXT: lwa 3, 4(4)
    // AIX-NEXT: blr
    ap.next_arg()
}

#[unsafe(no_mangle)]
unsafe extern "C" fn read_i64(ap: &mut VaList<'_>) -> i64 {
    // CHECK-LABEL: read_i64
    //
    // POWERPC: mr 5, 3
    // POWERPC-NEXT: lbz 3, 0(3)
    // POWERPC-NEXT: addi 3, 3, 1
    // POWERPC-NEXT: clrlwi 4, 3, 24
    // POWERPC-NEXT: cmplwi 4, 7
    // POWERPC-NEXT: bgt 0, .LBB2_2
    // POWERPC-NEXT: lwz 4, 8(5)
    // POWERPC-NEXT: rlwinm 6, 3, 0, 29, 30
    // POWERPC-NEXT: rlwinm 3, 3, 2, 27, 28
    // POWERPC-NEXT: addi 6, 6, 2
    // POWERPC-NEXT: add 4, 4, 3
    // POWERPC-NEXT: lwz 3, 0(4)
    // POWERPC-NEXT: lwz 4, 4(4)
    // POWERPC-NEXT: stb 6, 0(5)
    // POWERPC-NEXT: blr
    //
    // POWERPC64: ld 5, 0(3)
    // POWERPC64-NEXT: mr 4, 3
    // POWERPC64-NEXT: ld 3, 0(5)
    // POWERPC64-NEXT: addi 5, 5, 8
    // POWERPC64-NEXT: std 5, 0(4)
    // POWERPC64-NEXT: blr
    //
    // POWERPC64LE: ld 4, 0(3)
    // POWERPC64LE-NEXT: addi 5, 4, 8
    // POWERPC64LE-NEXT: std 5, 0(3)
    // POWERPC64LE-NEXT: ld 3, 0(4)
    // POWERPC64LE-NEXT: blr
    //
    // AIX: ld 4, 0(3)
    // AIX-NEXT: addi 5, 4, 8
    // AIX-NEXT: std 5, 0(3)
    // AIX-NEXT: ld 3, 0(4)
    // AIX-NEXT: blr
    ap.next_arg()
}

#[unsafe(no_mangle)]
#[cfg(target_pointer_width = "64")]
unsafe extern "C" fn read_i128(ap: &mut VaList<'_>) -> i128 {
    // POWERPC64-LABEL: read_i128
    // POWERPC64: ld 6, 0(3)
    // POWERPC64-NEXT: mr  5, 3
    // POWERPC64-NEXT: ld 3, 0(6)
    // POWERPC64-NEXT: ld 4, 8(6)
    // POWERPC64-NEXT: addi 6, 6, 16
    // POWERPC64-NEXT: std 6, 0(5)
    // POWERPC64-NEXT: blr
    //
    // POWERPC64LE-LABEL: read_i128
    // POWERPC64LE: ld 4, 0(3)
    // POWERPC64LE-NEXT: addi 5, 4, 16
    // POWERPC64LE-NEXT: std 5, 0(3)
    // POWERPC64LE-NEXT: ld 3, 0(4)
    // POWERPC64LE-NEXT: ld 4, 8(4)
    // POWERPC64LE-NEXT: blr
    //
    // AIX-LABEL: read_i128
    // AIX: ld 4, 0(3)
    // AIX-NEXT: addi 5, 4, 16
    // AIX-NEXT: std 5, 0(3)
    // AIX-NEXT: ld 3, 0(4)
    // AIX-NEXT: ld 4, 8(4)
    // AIX-NEXT: blr
    ap.next_arg()
}

#[unsafe(no_mangle)]
#[cfg(target_feature = "vsx")]
unsafe extern "C" fn read_f128(ap: &mut VaList<'_>) -> f128 {
    // POWERPC64LE-LABEL: read_f128
    // POWERPC64LE: ld 4, 0(3)
    // POWERPC64LE-NEXT: addi 4, 4, 15
    // POWERPC64LE-NEXT: rldicr 4, 4, 0, 59
    // POWERPC64LE-NEXT: lxvd2x 0, 0, 4
    // POWERPC64LE-NEXT: addi 4, 4, 16
    // POWERPC64LE-NEXT: std 4, 0(3)
    // POWERPC64LE-NEXT: xxswapd 34, 0
    // POWERPC64LE-NEXT: blr
    //
    // AIX-LABEL: read_f128
    // AIX: ld 4, 0(3)
    // AIX-NEXT: addi 4, 4, 15
    // AIX-NEXT: rldicr 4, 4, 0, 59
    // AIX-NEXT: lxvw4x 34, 0, 4
    // AIX-NEXT: addi 5, 4, 16
    // AIX-NEXT: std 5, 0(3)
    // AIX-NEXT: blr
    ap.next_arg()
}

#[unsafe(no_mangle)]
unsafe extern "C" fn read_ppcf128(ap: &mut VaList<'_>) -> ppcf128 {
    // CHECK-LABEL: read_ppcf128
    //
    // POWERPC: stwu 1, -32(1)
    // POWERPC: lbz 5, 1(3)
    // POWERPC-NEXT: cmplwi 5, 6
    // POWERPC-NEXT: bgt 0, .LBB3_2
    // POWERPC-NEXT: lwz 4, 8(3)
    // POWERPC-NEXT: rlwinm 6, 5, 3, 24, 28
    // POWERPC-NEXT: addi 5, 5, 2
    // POWERPC-NEXT: add 4, 4, 6
    // POWERPC-NEXT: addi 4, 4, 32
    // POWERPC-NEXT: b .LBB3_3
    // POWERPC-NEXT: .LBB3_2:
    // POWERPC-NEXT: lwz 4, 4(3)
    // POWERPC-NEXT: addi 4, 4, 7
    // POWERPC-NEXT: rlwinm 4, 4, 0, 0, 28
    // POWERPC-NEXT: addi 5, 4, 16
    // POWERPC-NEXT: stw 5, 4(3)
    // POWERPC-NEXT: li 5, 8
    // POWERPC-NEXT: .LBB3_3:
    // POWERPC-NEXT: lwz 6, 4(4)
    // POWERPC-NEXT: stb 5, 1(3)
    // POWERPC-NEXT: stw 6, 28(1)
    // POWERPC-NEXT: lwz 3, 0(4)
    // POWERPC-NEXT: stw 3, 24(1)
    // POWERPC-NEXT: lwz 3, 12(4)
    // POWERPC-NEXT: lfd 1, 24(1)
    // POWERPC-NEXT: stw 3, 20(1)
    // POWERPC-NEXT: lwz 3, 8(4)
    // POWERPC-NEXT: stw 3, 16(1)
    // POWERPC-NEXT: lfd 2, 16(1)
    // POWERPC-NEXT: addi 1, 1, 32
    // POWERPC-NEXT: blr
    //
    // POWERPC64: ld 4, 0(3)
    // POWERPC64-NEXT: lfd 1, 0(4)
    // POWERPC64-NEXT: lfd 2, 8(4)
    // POWERPC64-NEXT: addi 4, 4, 16
    // POWERPC64-NEXT: std 4, 0(3)
    // POWERPC64-NEXT: blr
    //
    // POWERPC64LE: ld 4, 0(3)
    // POWERPC64LE-NEXT: lfd 1, 0(4)
    // POWERPC64LE-NEXT: lfd 2, 8(4)
    // POWERPC64LE-NEXT: addi 5, 4, 16
    // POWERPC64LE-NEXT: std 5, 0(3)
    // POWERPC64LE-NEXT: blr
    //
    // AIX: ld 4, 0(3)
    // AIX-NEXT: lfd 1, 0(4)
    // AIX-NEXT: lfd 2, 8(4)
    // AIX-NEXT: addi 5, 4, 16
    // AIX-NEXT: std 5, 0(3)
    // AIX-NEXT: blr
    ap.next_arg()
}

#[unsafe(no_mangle)]
unsafe extern "C" fn read_ptr(ap: &mut VaList<'_>) -> *const u8 {
    // POWERPC: read_ptr = read_i32
    // POWERPC64: read_ptr = read_i64
    // POWERPC64LE: read_ptr = read_i64
    ap.next_arg()
}
