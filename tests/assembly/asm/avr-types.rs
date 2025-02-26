//@ add-core-stubs
//@ assembly-output: emit-asm
//@ compile-flags: --target avr-none -C target-cpu=atmega328p
//@ needs-llvm-components: avr

#![feature(no_core, asm_experimental_arch)]
#![crate_type = "rlib"]
#![no_core]
#![allow(non_camel_case_types)]

extern crate minicore;
use minicore::*;

type ptr = *const u64;

macro_rules! check {
    ($func:ident $ty:ident $class:ident) => {
        #[no_mangle]
        pub unsafe fn $func(x: $ty) -> $ty {
            let y;
            asm!("mov {}, {}", lateout($class) y, in($class) x);
            y
        }
    };
}

macro_rules! checkw {
    ($func:ident $ty:ident $class:ident) => {
        #[no_mangle]
        pub unsafe fn $func(x: $ty) -> $ty {
            let y;
            asm!("movw {}, {}", lateout($class) y, in($class) x);
            y
        }
    };
}

macro_rules! check_reg {
    ($func:ident $ty:ident $reg:tt) => {
        #[no_mangle]
        pub unsafe fn $func(x: $ty) -> $ty {
            let y;
            asm!(concat!("mov ", $reg, ", ", $reg), lateout($reg) y, in($reg) x);
            y
        }
    };
}

macro_rules! check_regw {
    ($func:ident $ty:ident $reg:tt $reg_lit:tt) => {
        #[no_mangle]
        pub unsafe fn $func(x: $ty) -> $ty {
            let y;
            asm!(concat!("movw ", $reg_lit, ", ", $reg_lit), lateout($reg) y, in($reg) x);
            y
        }
    };
}

extern "C" {
    fn extern_func();
    static extern_static: i8;
}

// CHECK-LABEL: sym_fn
// CHECK: ;APP
// CHECK: call extern_func
// CHECK: ;NO_APP
#[no_mangle]
pub unsafe fn sym_fn() {
    asm!("call {}", sym extern_func);
}

// CHECK-LABEL: sym_static
// CHECK: ;APP
// CHECK: lds r{{[0-9]+}}, extern_static
// CHECK: ;NO_APP
#[no_mangle]
pub unsafe fn sym_static() -> i8 {
    let y;
    asm!("lds {}, {}", lateout(reg) y, sym extern_static);
    y
}

// CHECK-LABEL: ld_z:
// CHECK: ;APP
// CHECK: ld r{{[0-9]+}}, Z
// CHECK: ;NO_APP
#[no_mangle]
pub unsafe fn ld_z(x: i16) -> i8 {
    let y;
    asm!("ld {}, Z", out(reg) y, in("Z") x);
    y
}

// CHECK-LABEL: ldd_z:
// CHECK: ;APP
// CHECK: ldd r{{[0-9]+}}, Z+4
// CHECK: ;NO_APP
#[no_mangle]
pub unsafe fn ldd_z(x: i16) -> i8 {
    let y;
    asm!("ldd {}, Z+4", out(reg) y, in("Z") x);
    y
}

// CHECK-LABEL: ld_predecrement:
// CHECK: ;APP
// CHECK: ld r{{[0-9]+}}, -Z
// CHECK: ;NO_APP
#[no_mangle]
pub unsafe fn ld_predecrement(x: i16) -> i8 {
    let y;
    asm!("ld {}, -Z", out(reg) y, in("Z") x);
    y
}

// CHECK-LABEL: ld_postincrement:
// CHECK: ;APP
// CHECK: ld r{{[0-9]+}}, Z+
// CHECK: ;NO_APP
#[no_mangle]
pub unsafe fn ld_postincrement(x: i16) -> i8 {
    let y;
    asm!("ld {}, Z+", out(reg) y, in("Z") x);
    y
}

// CHECK-LABEL: muls_clobber:
// CHECK: ;APP
// CHECK: muls r{{[0-9]+}}, r{{[0-9]+}}
// CHECK: movw r{{[0-9]+}}, r0
// CHECK: ;NO_APP
#[no_mangle]
pub unsafe fn muls_clobber(x: i8, y: i8) -> i16 {
    let z;
    asm!(
        "muls {}, {}",
        "movw {}, r1:r0",
        out(reg_iw) z,
        in(reg) x,
        in(reg) y,
    );
    z
}

// CHECK-LABEL: reg_i8:
// CHECK: ;APP
// CHECK: mov r{{[0-9]+}}, r{{[0-9]+}}
// CHECK: ;NO_APP
check!(reg_i8 i8 reg);

// CHECK-LABEL: reg_upper_i8:
// CHECK: ;APP
// CHECK: mov r{{[1-3][0-9]}}, r{{[1-3][0-9]}}
// CHECK: ;NO_APP
check!(reg_upper_i8 i8 reg_upper);

// CHECK-LABEL: reg_pair_i16:
// CHECK: ;APP
// CHECK: movw r{{[0-9]+}}, r{{[0-9]+}}
// CHECK: ;NO_APP
checkw!(reg_pair_i16 i16 reg_pair);

// CHECK-LABEL: reg_iw_i16:
// CHECK: ;APP
// CHECK: movw r{{[0-9]+}}, r{{[0-9]+}}
// CHECK: ;NO_APP
checkw!(reg_iw_i16 i16 reg_iw);

// CHECK-LABEL: reg_ptr_i16:
// CHECK: ;APP
// CHECK: movw r{{[0-9]+}}, r{{[0-9]+}}
// CHECK: ;NO_APP
checkw!(reg_ptr_i16 i16 reg_ptr);

// CHECK-LABEL: r2_i8:
// CHECK: ;APP
// CHECK: mov r2, r2
// CHECK: ;NO_APP
check_reg!(r2_i8 i8 "r2");

// CHECK-LABEL: xl_i8:
// CHECK: ;APP
// CHECK: mov r26, r26
// CHECK: ;NO_APP
check_reg!(xl_i8 i8 "XL");

// CHECK-LABEL: xh_i8:
// CHECK: ;APP
// CHECK: mov r27, r27
// CHECK: ;NO_APP
check_reg!(xh_i8 i8 "XH");

// CHECK-LABEL: x_i16:
// CHECK: ;APP
// CHECK: movw r26, r26
// CHECK: ;NO_APP
check_regw!(x_i16 i16 "X" "X");

// CHECK-LABEL: r25r24_i16:
// CHECK: ;APP
// CHECK: movw r24, r24
// CHECK: ;NO_APP
check_regw!(r25r24_i16 i16 "r25r24" "r24");
