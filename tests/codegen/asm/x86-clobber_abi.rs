//@ compile-flags: -Copt-level=3
//@ only-x86_64

#![crate_type = "rlib"]

use std::arch::asm;

// CHECK-LABEL: @clobber_sysv64
// CHECK: ={ax},={cx},={dx},={si},={di},={r8},={r9},={r10},={r11},={xmm0},={xmm1},={xmm2},={xmm3},={xmm4},={xmm5},={xmm6},={xmm7},={xmm8},={xmm9},={xmm10},={xmm11},={xmm12},={xmm13},={xmm14},={xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{k0},~{k1},~{k2},~{k3},~{k4},~{k5},~{k6},~{k7},~{st},~{st(1)},~{st(2)},~{st(3)},~{st(4)},~{st(5)},~{st(6)},~{st(7)},~{tmm0},~{tmm1},~{tmm2},~{tmm3},~{tmm4},~{tmm5},~{tmm6},~{tmm7},~{dirflag},~{fpsr},~{flags},~{memory}
#[no_mangle]
pub unsafe fn clobber_sysv64() {
    asm!("", clobber_abi("sysv64"));
}

// CHECK-LABEL: @clobber_win64
// CHECK: ={ax},={cx},={dx},={r8},={r9},={r10},={r11},={xmm0},={xmm1},={xmm2},={xmm3},={xmm4},={xmm5},={xmm6},={xmm7},={xmm8},={xmm9},={xmm10},={xmm11},={xmm12},={xmm13},={xmm14},={xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{k0},~{k1},~{k2},~{k3},~{k4},~{k5},~{k6},~{k7},~{st},~{st(1)},~{st(2)},~{st(3)},~{st(4)},~{st(5)},~{st(6)},~{st(7)},~{tmm0},~{tmm1},~{tmm2},~{tmm3},~{tmm4},~{tmm5},~{tmm6},~{tmm7},~{dirflag},~{fpsr},~{flags},~{memory}
#[no_mangle]
pub unsafe fn clobber_win64() {
    asm!("", clobber_abi("win64"));
}

// CHECK-LABEL: @clobber_sysv64
// CHECK: =&{dx},={ax},={cx},={si},={di},={r8},={r9},={r10},={r11},={xmm0},={xmm1},={xmm2},={xmm3},={xmm4},={xmm5},={xmm6},={xmm7},={xmm8},={xmm9},={xmm10},={xmm11},={xmm12},={xmm13},={xmm14},={xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{k0},~{k1},~{k2},~{k3},~{k4},~{k5},~{k6},~{k7},~{st},~{st(1)},~{st(2)},~{st(3)},~{st(4)},~{st(5)},~{st(6)},~{st(7)},~{tmm0},~{tmm1},~{tmm2},~{tmm3},~{tmm4},~{tmm5},~{tmm6},~{tmm7},~{dirflag},~{fpsr},~{flags},~{memory}
#[no_mangle]
pub unsafe fn clobber_sysv64_edx() {
    let foo: i32;
    asm!("", out("edx") foo, clobber_abi("sysv64"));
}

// CHECK-LABEL: @clobber_win64
// CHECK: =&{dx},={ax},={cx},={r8},={r9},={r10},={r11},={xmm0},={xmm1},={xmm2},={xmm3},={xmm4},={xmm5},={xmm6},={xmm7},={xmm8},={xmm9},={xmm10},={xmm11},={xmm12},={xmm13},={xmm14},={xmm15},~{xmm16},~{xmm17},~{xmm18},~{xmm19},~{xmm20},~{xmm21},~{xmm22},~{xmm23},~{xmm24},~{xmm25},~{xmm26},~{xmm27},~{xmm28},~{xmm29},~{xmm30},~{xmm31},~{k0},~{k1},~{k2},~{k3},~{k4},~{k5},~{k6},~{k7},~{st},~{st(1)},~{st(2)},~{st(3)},~{st(4)},~{st(5)},~{st(6)},~{st(7)},~{tmm0},~{tmm1},~{tmm2},~{tmm3},~{tmm4},~{tmm5},~{tmm6},~{tmm7},~{dirflag},~{fpsr},~{flags},~{memory}
#[no_mangle]
pub unsafe fn clobber_win64_edx() {
    let foo: i32;
    asm!("", out("edx") foo, clobber_abi("win64"));
}
