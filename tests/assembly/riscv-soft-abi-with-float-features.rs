//@ assembly-output: emit-asm
//@ compile-flags: --target riscv64imac-unknown-none-elf -Ctarget-feature=+f,+d
//@ needs-llvm-components: riscv

#![feature(no_core, lang_items, f16)]
#![crate_type = "lib"]
#![no_core]

#[lang = "sized"]
trait Sized {}

#[lang = "copy"]
trait Copy {}

impl Copy for f16 {}
impl Copy for f32 {}
impl Copy for f64 {}

// This test checks that the floats are all returned in `a0` as required by the `lp64` ABI.

// CHECK-LABEL: read_f16
#[no_mangle]
pub extern "C" fn read_f16(x: &f16) -> f16 {
    // CHECK: lh a0, 0(a0)
    // CHECK-NEXT: lui a1, 1048560
    // CHECK-NEXT: or a0, a0, a1
    // CHECK-NEXT: ret
    *x
}

// CHECK-LABEL: read_f32
#[no_mangle]
pub extern "C" fn read_f32(x: &f32) -> f32 {
    // CHECK: flw fa5, 0(a0)
    // CHECK-NEXT: fmv.x.w a0, fa5
    // CHECK-NEXT: ret
    *x
}

// CHECK-LABEL: read_f64
#[no_mangle]
pub extern "C" fn read_f64(x: &f64) -> f64 {
    // CHECK: ld a0, 0(a0)
    // CHECK-NEXT: ret
    *x
}
