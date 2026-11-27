//@ add-minicore
//@ revisions: esp32 esp32s2 call0
//@[esp32] compile-flags: --target xtensa-esp32-none-elf
//@[esp32s2] compile-flags: --target xtensa-esp32s2-none-elf
//@[call0] compile-flags: --target xtensa-esp32-none-elf -Ctarget-cpu=generic
//@ needs-llvm-components: xtensa
// ignore-tidy-linelength

#![crate_type = "rlib"]
#![feature(no_core, asm_experimental_arch)]
#![no_core]

extern crate minicore;
use minicore::*;

// CHECK-LABEL: @no_clobber
// CHECK: call void asm sideeffect "", ""()
#[no_mangle]
pub unsafe fn no_clobber() {
    asm!("", options(nostack, nomem, preserves_flags));
}

// The registers of missing target features are allowed as clobbers.
// CHECK-LABEL: @optional_clobbers
// esp32: asm sideeffect "", "=&{f0},~{b0},~{lbeg},~{acclo},~{scompare1}"()
// esp32s2: asm sideeffect "", "~{f0},~{b0},~{lbeg},~{acclo},~{scompare1}"()
// call0: asm sideeffect "", "~{f0},~{b0},~{lbeg},~{acclo},~{scompare1}"()
#[no_mangle]
pub unsafe fn optional_clobbers() {
    asm!(
        "",
        out("f0") _,
        out("b0") _,
        out("lbeg") _,
        out("acclo") _,
        out("scompare1") _,
        options(nostack, nomem, preserves_flags)
    );
}

// CHECK-LABEL: @clobber_abi
// esp32: asm sideeffect "", "={a8},={a9},={a10},={a11},={a12},={a13},={a14},={a15},={f0},={f1},={f2},={f3},={f4},={f5},={f6},={f7},={f8},={f9},={f10},={f11},={f12},={f13},={f14},={f15},~{sar},~{scompare1},~{lbeg},~{lend},~{lcount},~{acclo},~{acchi},~{m0},~{m1},~{m2},~{m3},~{b0},~{b1},~{b2},~{b3},~{b4},~{b5},~{b6},~{b7},~{b8},~{b9},~{b10},~{b11},~{b12},~{b13},~{b14},~{b15}"()
// esp32s2: asm sideeffect "", "={a8},={a9},={a10},={a11},={a12},={a13},={a14},={a15},~{f0},~{f1},~{f2},~{f3},~{f4},~{f5},~{f6},~{f7},~{f8},~{f9},~{f10},~{f11},~{f12},~{f13},~{f14},~{f15},~{sar},~{scompare1},~{lbeg},~{lend},~{lcount},~{acclo},~{acchi},~{m0},~{m1},~{m2},~{m3},~{b0},~{b1},~{b2},~{b3},~{b4},~{b5},~{b6},~{b7},~{b8},~{b9},~{b10},~{b11},~{b12},~{b13},~{b14},~{b15}"()
// call0: asm sideeffect "", "={a2},={a3},={a4},={a5},={a6},={a7},={a8},={a9},={a10},={a11},~{f0},~{f1},~{f2},~{f3},~{f4},~{f5},~{f6},~{f7},~{f8},~{f9},~{f10},~{f11},~{f12},~{f13},~{f14},~{f15},~{sar},~{scompare1},~{lbeg},~{lend},~{lcount},~{acclo},~{acchi},~{m0},~{m1},~{m2},~{m3},~{b0},~{b1},~{b2},~{b3},~{b4},~{b5},~{b6},~{b7},~{b8},~{b9},~{b10},~{b11},~{b12},~{b13},~{b14},~{b15}"()
#[no_mangle]
pub unsafe fn clobber_abi() {
    asm!("", clobber_abi("C"), options(nostack, nomem, preserves_flags));
}
