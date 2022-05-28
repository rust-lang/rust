// min-llvm-version: 13.0
// assembly-output: emit-asm
// compile-flags: --target msp430-none-elf
// needs-llvm-components: msp430

#![feature(no_core, lang_items, rustc_attrs, asm_sym, asm_experimental_arch, asm_const)]
#![crate_type = "rlib"]
#![no_core]
#![allow(non_camel_case_types)]

#[rustc_builtin_macro]
macro_rules! asm {
    () => {};
}
#[rustc_builtin_macro]
macro_rules! concat {
    () => {};
}

#[lang = "sized"]
trait Sized {}
#[lang = "copy"]
trait Copy {}

type ptr = *const i16;

impl Copy for i8 {}
impl Copy for i16 {}
impl Copy for i32 {}
impl Copy for i64 {}
impl Copy for ptr {}

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

macro_rules! checkb {
    ($func:ident $ty:ident $class:ident) => {
        #[no_mangle]
        pub unsafe fn $func(x: $ty) -> $ty {
            let y;
            asm!("mov.b {}, {}", lateout($class) y, in($class) x);
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

macro_rules! check_regb {
    ($func:ident $ty:ident $reg:tt) => {
        #[no_mangle]
        pub unsafe fn $func(x: $ty) -> $ty {
            let y;
            asm!(concat!("mov.b ", $reg, ", ", $reg), lateout($reg) y, in($reg) x);
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
// CHECK: mov.b extern_static, r{{[0-9]+}}
// CHECK: ;NO_APP
#[no_mangle]
pub unsafe fn sym_static() -> i8 {
    let y;
    asm!("mov.b {1}, {0}", lateout(reg) y, sym extern_static);
    y
}

// CHECK-LABEL: add_const:
// CHECK: ;APP
// CHECK: add.b #5, r{{[0-9]+}}
// CHECK: ;NO_APP
#[no_mangle]
pub unsafe fn add_const() -> i8 {
    let y;
    asm!("add.b #{number}, {}", out(reg) y, number = const 5);
    y
}

// CHECK-LABEL: mov_postincrement:
// CHECK: ;APP
// CHECK: mov @r5+, r{{[0-9]+}}
// CHECK: ;NO_APP
#[no_mangle]
pub unsafe fn mov_postincrement(mut x: *const i16) -> (i16, *const i16) {
    let y;
    asm!("mov @r5+, {0}", out(reg) y, inlateout("r5") x);
    (y, x)
}

// CHECK-LABEL: reg_i8:
// CHECK: ;APP
// CHECK: mov r{{[0-9]+}}, r{{[0-9]+}}
// CHECK: ;NO_APP
check!(reg_i8 i8 reg);

// CHECK-LABEL: reg_i16:
// CHECK: ;APP
// CHECK: mov r{{[0-9]+}}, r{{[0-9]+}}
// CHECK: ;NO_APP
check!(reg_i16 i16 reg);

// CHECK-LABEL: reg_i8b:
// CHECK: ;APP
// CHECK: mov.b r{{[0-9]+}}, r{{[0-9]+}}
// CHECK: ;NO_APP
checkb!(reg_i8b i8 reg);

// CHECK-LABEL: r5_i8:
// CHECK: ;APP
// CHECK: mov r5, r5
// CHECK: ;NO_APP
check_reg!(r5_i8 i8 "r5");

// CHECK-LABEL: r5_i16:
// CHECK: ;APP
// CHECK: mov r5, r5
// CHECK: ;NO_APP
check_reg!(r5_i16 i16 "r5");

// CHECK-LABEL: r5_i8b:
// CHECK: ;APP
// CHECK: mov.b r5, r5
// CHECK: ;NO_APP
check_regb!(r5_i8b i8 "r5");
