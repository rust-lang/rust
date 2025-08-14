//@ add-core-stubs
//@ revisions: loongarch32 loongarch64

//@ assembly-output: emit-asm

//@[loongarch32] compile-flags: --target loongarch32-unknown-none
//@[loongarch32] needs-llvm-components: loongarch

//@[loongarch64] compile-flags: --target loongarch64-unknown-none
//@[loongarch64] needs-llvm-components: loongarch

//@ compile-flags: -Zmerge-functions=disabled

#![feature(no_core, f16)]
#![crate_type = "rlib"]
#![no_core]
#![allow(asm_sub_register, non_camel_case_types)]

extern crate minicore;
use minicore::*;

type ptr = *const i32;

extern "C" {
    fn extern_func();
    static extern_static: u8;
}

// CHECK-LABEL: sym_fn:
// CHECK: #APP
// CHECK: pcalau12i $t0, %got_pc_hi20(extern_func)
// CHECK: ld.{{[wd]}} $t0, $t0, %got_pc_lo12(extern_func)
// CHECK: #NO_APP
#[no_mangle]
pub unsafe fn sym_fn() {
    asm!("la.got $r12, {}", sym extern_func);
}

// CHECK-LABEL: sym_static:
// CHECK: #APP
// CHECK: pcalau12i $t0, %got_pc_hi20(extern_static)
// CHECK: ld.{{[wd]}} $t0, $t0, %got_pc_lo12(extern_static)
// CHECK: #NO_APP
#[no_mangle]
pub unsafe fn sym_static() {
    asm!("la.got $r12, {}", sym extern_static);
}

macro_rules! check { ($func:ident, $ty:ty, $class:ident, $mov:literal) => {
    #[no_mangle]
    pub unsafe fn $func(x: $ty) -> $ty {
        let y;
        asm!(concat!($mov," {}, {}"), out($class) y, in($class) x);
        y
    }
};}

macro_rules! check_reg { ($func:ident, $ty:ty, $reg:tt, $mov:literal) => {
    #[no_mangle]
    pub unsafe fn $func(x: $ty) -> $ty {
        let y;
        asm!(concat!($mov, " ", $reg, ", ", $reg), lateout($reg) y, in($reg) x);
        y
    }
};}

// CHECK-LABEL: reg_i8:
// CHECK: #APP
// CHECK: move ${{[a-z0-9]+}}, ${{[a-z0-9]+}}
// CHECK: #NO_APP
check!(reg_i8, i8, reg, "move");

// CHECK-LABEL: reg_i16:
// CHECK: #APP
// CHECK: move ${{[a-z0-9]+}}, ${{[a-z0-9]+}}
// CHECK: #NO_APP
check!(reg_i16, i16, reg, "move");

// CHECK-LABEL: reg_f16:
// CHECK: #APP
// CHECK: move ${{[a-z0-9]+}}, ${{[a-z0-9]+}}
// CHECK: #NO_APP
check!(reg_f16, f16, reg, "move");

// CHECK-LABEL: reg_i32:
// CHECK: #APP
// CHECK: move ${{[a-z0-9]+}}, ${{[a-z0-9]+}}
// CHECK: #NO_APP
check!(reg_i32, i32, reg, "move");

// CHECK-LABEL: reg_f32:
// CHECK: #APP
// CHECK: move ${{[a-z0-9]+}}, ${{[a-z0-9]+}}
// CHECK: #NO_APP
check!(reg_f32, f32, reg, "move");

// loongarch64-LABEL: reg_i64:
// loongarch64: #APP
// loongarch64: move ${{[a-z0-9]+}}, ${{[a-z0-9]+}}
// loongarch64: #NO_APP
#[cfg(loongarch64)]
check!(reg_i64, i64, reg, "move");

// loongarch64-LABEL: reg_f64:
// loongarch64: #APP
// loongarch64: move ${{[a-z0-9]+}}, ${{[a-z0-9]+}}
// loongarch64: #NO_APP
#[cfg(loongarch64)]
check!(reg_f64, f64, reg, "move");

// CHECK-LABEL: reg_ptr:
// CHECK: #APP
// CHECK: move ${{[a-z0-9]+}}, ${{[a-z0-9]+}}
// CHECK: #NO_APP
check!(reg_ptr, ptr, reg, "move");

// CHECK-LABEL: freg_f16:
// CHECK: #APP
// CHECK: fmov.s $f{{[a-z0-9]+}}, $f{{[a-z0-9]+}}
// CHECK: #NO_APP
check!(freg_f16, f16, freg, "fmov.s");

// CHECK-LABEL: freg_f32:
// CHECK: #APP
// CHECK: fmov.s $f{{[a-z0-9]+}}, $f{{[a-z0-9]+}}
// CHECK: #NO_APP
check!(freg_f32, f32, freg, "fmov.s");

// CHECK-LABEL: freg_f64:
// CHECK: #APP
// CHECK: fmov.d $f{{[a-z0-9]+}}, $f{{[a-z0-9]+}}
// CHECK: #NO_APP
check!(freg_f64, f64, freg, "fmov.d");

// CHECK-LABEL: r4_i8:
// CHECK: #APP
// CHECK: move $a0, $a0
// CHECK: #NO_APP
check_reg!(r4_i8, i8, "$r4", "move");

// CHECK-LABEL: r4_i16:
// CHECK: #APP
// CHECK: move $a0, $a0
// CHECK: #NO_APP
check_reg!(r4_i16, i16, "$r4", "move");

// CHECK-LABEL: r4_f16:
// CHECK: #APP
// CHECK: move $a0, $a0
// CHECK: #NO_APP
check_reg!(r4_f16, f16, "$r4", "move");

// CHECK-LABEL: r4_i32:
// CHECK: #APP
// CHECK: move $a0, $a0
// CHECK: #NO_APP
check_reg!(r4_i32, i32, "$r4", "move");

// CHECK-LABEL: r4_f32:
// CHECK: #APP
// CHECK: move $a0, $a0
// CHECK: #NO_APP
check_reg!(r4_f32, f32, "$r4", "move");

// loongarch64-LABEL: r4_i64:
// loongarch64: #APP
// loongarch64: move $a0, $a0
// loongarch64: #NO_APP
#[cfg(loongarch64)]
check_reg!(r4_i64, i64, "$r4", "move");

// loongarch64-LABEL: r4_f64:
// loongarch64: #APP
// loongarch64: move $a0, $a0
// loongarch64: #NO_APP
#[cfg(loongarch64)]
check_reg!(r4_f64, f64, "$r4", "move");

// CHECK-LABEL: r4_ptr:
// CHECK: #APP
// CHECK: move $a0, $a0
// CHECK: #NO_APP
check_reg!(r4_ptr, ptr, "$r4", "move");

// CHECK-LABEL: f0_f16:
// CHECK: #APP
// CHECK: fmov.s $f{{[a-z0-9]+}}, $f{{[a-z0-9]+}}
// CHECK: #NO_APP
check_reg!(f0_f16, f16, "$f0", "fmov.s");

// CHECK-LABEL: f0_f32:
// CHECK: #APP
// CHECK: fmov.s $f{{[a-z0-9]+}}, $f{{[a-z0-9]+}}
// CHECK: #NO_APP
check_reg!(f0_f32, f32, "$f0", "fmov.s");

// CHECK-LABEL: f0_f64:
// CHECK: #APP
// CHECK: fmov.d $f{{[a-z0-9]+}}, $f{{[a-z0-9]+}}
// CHECK: #NO_APP
check_reg!(f0_f64, f64, "$f0", "fmov.d");
