//@ add-core-stubs
//@ assembly-output: emit-asm
//@ compile-flags: -Copt-level=3 --target riscv64gc-unknown-linux-gnu
//@ needs-llvm-components: riscv

#![feature(no_core, lang_items)]
#![no_std]
#![no_core]
#![crate_type = "lib"]

extern crate minicore;
use minicore::*;

#[repr(C, align(64))]
struct Aligned(f64);

#[repr(C)]
struct Padded(u8, Aligned);

#[repr(C, packed)]
struct Packed(u8, f32);

impl Copy for Aligned {}
impl Copy for Padded {}
impl Copy for Packed {}

extern "C" {
    fn take_padded(x: Padded);
    fn get_padded() -> Padded;
    fn take_packed(x: Packed);
    fn get_packed() -> Packed;
}

// CHECK-LABEL: pass_padded
#[unsafe(no_mangle)]
extern "C" fn pass_padded(out: &mut Padded, x: Padded) {
    // CHECK: sb a1, 0(a0)
    // CHECK-NEXT: fsd fa0, 64(a0)
    // CHECK-NEXT: ret
    *out = x;
}

// CHECK-LABEL: ret_padded
#[unsafe(no_mangle)]
extern "C" fn ret_padded(x: &Padded) -> Padded {
    // CHECK: fld fa0, 64(a0)
    // CHECK-NEXT: lbu a0, 0(a0)
    // CHECK-NEXT: ret
    *x
}

#[unsafe(no_mangle)]
extern "C" fn call_padded(x: &Padded) {
    // CHECK: fld fa0, 64(a0)
    // CHECK-NEXT: lbu a0, 0(a0)
    // CHECK-NEXT: tail take_padded
    unsafe {
        take_padded(*x);
    }
}

#[unsafe(no_mangle)]
extern "C" fn receive_padded(out: &mut Padded) {
    // CHECK: addi sp, sp, -16
    // CHECK-NEXT: .cfi_def_cfa_offset 16
    // CHECK-NEXT: sd ra, [[#%d,RA_SPILL:]](sp)
    // CHECK-NEXT: sd [[TEMP:.*]], [[#%d,TEMP_SPILL:]](sp)
    // CHECK-NEXT: .cfi_offset ra, [[#%d,RA_SPILL - 16]]
    // CHECK-NEXT: .cfi_offset [[TEMP]], [[#%d,TEMP_SPILL - 16]]
    // CHECK-NEXT: mv [[TEMP]], a0
    // CHECK-NEXT: call get_padded
    // CHECK-NEXT: sb a0, 0([[TEMP]])
    // CHECK-NEXT: fsd fa0, 64([[TEMP]])
    // CHECK-NEXT: ld ra, [[#%d,RA_SPILL]](sp)
    // CHECK-NEXT: ld [[TEMP]], [[#%d,TEMP_SPILL]](sp)
    // CHECK: addi sp, sp, 16
    // CHECK: ret
    unsafe {
        *out = get_padded();
    }
}

// CHECK-LABEL: pass_packed
#[unsafe(no_mangle)]
extern "C" fn pass_packed(out: &mut Packed, x: Packed) {
    // CHECK: addi sp, sp, -16
    // CHECK-NEXT: .cfi_def_cfa_offset 16
    // CHECK-NEXT: sb a1, 0(a0)
    // CHECK-NEXT: fsw fa0, 8(sp)
    // CHECK-NEXT: lw [[VALUE:.*]], 8(sp)
    // CHECK-DAG: srli [[BYTE4:.*]], [[VALUE]], 24
    // CHECK-DAG: srli [[BYTE3:.*]], [[VALUE]], 16
    // CHECK-DAG: srli [[BYTE2:.*]], [[VALUE]], 8
    // CHECK-DAG: sb [[VALUE]], 1(a0)
    // CHECK-DAG: sb [[BYTE2]], 2(a0)
    // CHECK-DAG: sb [[BYTE3]], 3(a0)
    // CHECK-DAG: sb [[BYTE4]], 4(a0)
    // CHECK-NEXT: addi sp, sp, 16
    // CHECK: ret
    *out = x;
}

// CHECK-LABEL: ret_packed
#[unsafe(no_mangle)]
extern "C" fn ret_packed(x: &Packed) -> Packed {
    // CHECK: addi sp, sp, -16
    // CHECK-NEXT: .cfi_def_cfa_offset 16
    // CHECK-NEXT: lbu [[BYTE2:.*]], 2(a0)
    // CHECK-NEXT: lbu [[BYTE1:.*]], 1(a0)
    // CHECK-NEXT: lbu [[BYTE3:.*]], 3(a0)
    // CHECK-NEXT: lbu [[BYTE4:.*]], 4(a0)
    // CHECK-NEXT: slli [[SHIFTED2:.*]], [[BYTE2]], 8
    // CHECK-NEXT: or [[BYTE12:.*]], [[SHIFTED2]], [[BYTE1]]
    // CHECK-NEXT: slli [[SHIFTED3:.*]], [[BYTE3]], 16
    // CHECK-NEXT: slli [[SHIFTED4:.*]], [[BYTE4]], 24
    // CHECK-NEXT: or [[BYTE34:.*]], [[SHIFTED3]], [[SHIFTED4]]
    // CHECK-NEXT: or [[VALUE:.*]], [[BYTE12]], [[BYTE34]]
    // CHECK-NEXT: sw [[VALUE]], 8(sp)
    // CHECK-NEXT: flw fa0, 8(sp)
    // CHECK-NEXT: lbu a0, 0(a0)
    // CHECK-NEXT: addi sp, sp, 16
    // CHECK: ret
    *x
}

#[unsafe(no_mangle)]
extern "C" fn call_packed(x: &Packed) {
    // CHECK: addi sp, sp, -16
    // CHECK-NEXT: .cfi_def_cfa_offset 16
    // CHECK-NEXT: lbu [[BYTE2:.*]], 2(a0)
    // CHECK-NEXT: lbu [[BYTE1:.*]], 1(a0)
    // CHECK-NEXT: lbu [[BYTE3:.*]], 3(a0)
    // CHECK-NEXT: lbu [[BYTE4:.*]], 4(a0)
    // CHECK-NEXT: slli [[SHIFTED2:.*]], [[BYTE2]], 8
    // CHECK-NEXT: or [[BYTE12:.*]], [[SHIFTED2]], [[BYTE1]]
    // CHECK-NEXT: slli [[SHIFTED3:.*]], [[BYTE3]], 16
    // CHECK-NEXT: slli [[SHIFTED4:.*]], [[BYTE4]], 24
    // CHECK-NEXT: or [[BYTE34:.*]], [[SHIFTED3]], [[SHIFTED4]]
    // CHECK-NEXT: or [[VALUE:.*]], [[BYTE12]], [[BYTE34]]
    // CHECK-NEXT: sw [[VALUE]], 8(sp)
    // CHECK-NEXT: flw fa0, 8(sp)
    // CHECK-NEXT: lbu a0, 0(a0)
    // CHECK-NEXT: addi sp, sp, 16
    // CHECK: tail take_packed
    unsafe {
        take_packed(*x);
    }
}

#[unsafe(no_mangle)]
extern "C" fn receive_packed(out: &mut Packed) {
    // CHECK: addi sp, sp, -32
    // CHECK-NEXT: .cfi_def_cfa_offset 32
    // CHECK-NEXT: sd ra, [[#%d,RA_SPILL:]](sp)
    // CHECK-NEXT: sd [[TEMP:.*]], [[#%d,TEMP_SPILL:]](sp)
    // CHECK-NEXT: .cfi_offset ra, [[#%d,RA_SPILL - 32]]
    // CHECK-NEXT: .cfi_offset [[TEMP]], [[#%d,TEMP_SPILL - 32]]
    // CHECK-NEXT: mv [[TEMP]], a0
    // CHECK-NEXT: call get_packed
    // CHECK-NEXT: sb a0, 0([[TEMP]])
    // CHECK-NEXT: fsw fa0, [[FLOAT_SPILL:.*]](sp)
    // CHECK-NEXT: lw [[VALUE:.*]], [[FLOAT_SPILL]](sp)
    // CHECK-DAG: srli [[BYTE4:.*]], [[VALUE]], 24
    // CHECK-DAG: srli [[BYTE3:.*]], [[VALUE]], 16
    // CHECK-DAG: srli [[BYTE2:.*]], [[VALUE]], 8
    // CHECK-DAG: sb [[VALUE]], 1([[TEMP]])
    // CHECK-DAG: sb [[BYTE2]], 2([[TEMP]])
    // CHECK-DAG: sb [[BYTE3]], 3([[TEMP]])
    // CHECK-DAG: sb [[BYTE4]], 4([[TEMP]])
    // CHECK-NEXT: ld ra, [[#%d,RA_SPILL]](sp)
    // CHECK-NEXT: ld [[TEMP]], [[#%d,TEMP_SPILL]](sp)
    // CHECK: addi sp, sp, 32
    // CHECK: ret
    unsafe {
        *out = get_packed();
    }
}
