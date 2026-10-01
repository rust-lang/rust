//@ assembly-output: emit-asm
//@ add-minicore
//@ revisions: linux windows-gnu windows-msvc
//@[linux] compile-flags: --target i686-unknown-linux-gnu
//@[linux] needs-llvm-components: x86
//@[windows-gnu] compile-flags: --target i686-pc-windows-gnu
//@[windows-gnu] needs-llvm-components: x86
//@[windows-msvc] compile-flags: --target i686-pc-windows-msvc
//@[windows-msvc] needs-llvm-components: x86
//@ compile-flags: -Copt-level=3 -Cllvm-args=-x86-asm-syntax=intel

#![feature(no_core)]
#![no_core]
#![crate_type = "lib"]

extern crate minicore;
use minicore::*;

// Tests that returning `f32` and `f64` with the "C" ABI on 32-bit x86 preserves signalling NaNs.

// CHECK-LABEL: return_f32:
#[unsafe(no_mangle)]
pub extern "C" fn return_f32(x: f32) -> f32 {
    // CHECK: movss [[XMM:.*]], dword ptr [{{esp|ebp}} + [[#]]]
    // CHECK-NEXT: ucomiss [[XMM]], [[XMM]]
    // CHECK-NEXT: jp [[NAN_LABEL:.*]]
    // CHECK-NEXT: movss dword ptr [esp + [[#OFFSET:]]], [[XMM]]
    // CHECK-NEXT: fld dword ptr [esp + [[#OFFSET]]]
    // CHECK: ret
    // CHECK-NEXT: [[NAN_LABEL]]:
    // CHECK: movd [[BITS:.*]], [[XMM]]
    // CHECK-NEXT: mov dword ptr [esp + [[#OFFSET:]]], 0
    // CHECK-NEXT: mov e[[SIGN_AND_EXP:.*]], [[BITS]]
    // CHECK-NEXT: shl [[BITS]], 8
    // CHECK-NEXT: shr e[[SIGN_AND_EXP]], 16
    // CHECK-NEXT: mov dword ptr [esp + [[#OFFSET+4]]], [[BITS]]
    // CHECK-NEXT: or e[[SIGN_AND_EXP]], 32767
    // CHECK-NEXT: mov word ptr [esp + [[#OFFSET+8]]], [[SIGN_AND_EXP]]
    // CHECK-NEXT: fld tbyte ptr [esp + [[#OFFSET]]]
    // CHECK: ret
    x
}

// CHECK-LABEL: return_f64:
#[unsafe(no_mangle)]
pub extern "C" fn return_f64(x: f64) -> f64 {
    // CHECK: movsd [[XMM:.*]], qword ptr [{{esp|ebp}} + {{.*}}]
    // CHECK-NEXT: ucomisd [[XMM]], [[XMM]]
    // CHECK-NEXT: jp [[NAN_LABEL:.*]]
    // CHECK-NEXT: movsd qword ptr [esp + [[#OFFSET:]]], [[XMM]]
    // CHECK-NEXT: fld qword ptr [esp + [[#OFFSET]]]
    // CHECK: ret
    // CHECK-NEXT: [[NAN_LABEL]]:
    // CHECK: movsd qword ptr [esp + [[#OFFSET:]]], [[XMM]]
    // CHECK-NEXT: mov [[HIGH:.*]], dword ptr [esp + [[#OFFSET+4]]]
    // CHECK-NEXT: mov [[LOW:.*]], dword ptr [esp + [[#OFFSET]]]
    // CHECK-NEXT: mov e[[SIGN_AND_EXP:.*]], [[HIGH]]
    // CHECK-NEXT: shld [[HIGH]], [[LOW]], 11
    // CHECK-NEXT: shl [[LOW]], 11
    // CHECK-NEXT: shr e[[SIGN_AND_EXP]], 16
    // CHECK-NEXT: mov dword ptr [esp + 4], [[HIGH]]
    // CHECK-NEXT: mov dword ptr [esp], [[LOW]]
    // CHECK-NEXT: or e[[SIGN_AND_EXP]], 32767
    // CHECK-NEXT: mov word ptr [esp + 8], [[SIGN_AND_EXP]]
    // CHECK-NEXT: fld tbyte ptr [esp]
    // CHECK: ret
    x
}

// CHECK-LABEL: call_f32:
#[unsafe(no_mangle)]
pub unsafe fn call_f32(x: &mut f32) {
    extern "C" {
        fn get_f32() -> f32;
    }
    // CHECK: mov [[PTR:.*]], dword ptr [{{esp|ebp}} + {{.*}}]
    // CHECK: call {{()|_}}get_f32
    // CHECK-NEXT: fucomi st, st(0)
    // CHECK-NEXT: jp [[NAN_LABEL:.*]]
    // CHECK-NEXT: fstp dword ptr [esp + [[#OFFSET:]]]
    // CHECK-NEXT: movd [[XMM:.*]], dword ptr [esp + [[#OFFSET]]]
    // CHECK-NEXT: [[RET_LABEL:.*]]:
    // CHECK-NEXT: movd dword ptr [[[PTR]]], [[XMM]]
    // CHECK: ret
    // CHECK-NEXT: [[NAN_LABEL]]:
    // CHECK: fstp tbyte ptr [esp + [[#OFFSET:]]]
    // CHECK-NEXT: mov [[SIGN_AND_EXP:.*]], dword ptr [esp + [[#OFFSET+8]]]
    // CHECK-NEXT: mov [[BITS:.*]], dword ptr [esp + [[#OFFSET+4]]]
    // CHECK-NEXT: and [[SIGN_AND_EXP]], -128
    // CHECK-NEXT: shr [[BITS]], 8
    // CHECK-NEXT: shl [[SIGN_AND_EXP]], 16
    // CHECK-NEXT: or [[BITS]], [[SIGN_AND_EXP]]
    // CHECK-NEXT: movd [[XMM]], [[BITS]]
    // CHECK-NEXT: jmp [[RET_LABEL]]
    *x = get_f32();
}

// CHECK-LABEL: call_f64:
#[unsafe(no_mangle)]
pub unsafe fn call_f64(x: &mut f64) {
    extern "C" {
        fn get_f64() -> f64;
    }
    // CHECK: mov [[PTR:.*]], dword ptr [{{esp|ebp}} + {{.*}}]
    // CHECK: call {{()|_}}get_f64
    // CHECK-NEXT: fucomi st, st(0)
    // CHECK-NEXT: jp [[NAN_LABEL:.*]]
    // CHECK-NEXT: fstp qword ptr [esp + [[#OFFSET:]]]
    // CHECK-NEXT: movq [[XMM:.*]], qword ptr [esp + [[#OFFSET]]]
    // CHECK-NEXT: [[RET_LABEL:.*]]:
    // CHECK-NEXT: movq qword ptr [[[PTR]]], [[XMM]]
    // CHECK: ret
    // CHECK-NEXT: [[NAN_LABEL]]:
    // CHECK: fstp tbyte ptr [esp]
    // CHECK-NEXT: mov [[SIGN_AND_EXP:.*]], dword ptr [esp + 8]
    // CHECK-NEXT: mov [[LOW:.*]], dword ptr [esp]
    // CHECK-NEXT: mov [[HIGH:.*]], dword ptr [esp + 4]
    // CHECK-NEXT: and [[SIGN_AND_EXP]], -16
    // CHECK-NEXT: shrd [[LOW]], [[HIGH]], 11
    // CHECK-NEXT: shr [[HIGH]], 11
    // CHECK-NEXT: shl [[SIGN_AND_EXP]], 16
    // CHECK-NEXT: movd [[XMM]], [[LOW]]
    // CHECK-NEXT: or [[HIGH]], [[SIGN_AND_EXP]]
    // CHECK-NEXT: movd [[XMM_HIGH:.*]], [[HIGH]]
    // CHECK-NEXT: punpckldq [[XMM]], [[XMM_HIGH]]
    // CHECK-NEXT: jmp [[RET_LABEL]]
    *x = get_f64();
}
