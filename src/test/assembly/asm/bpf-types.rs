// min-llvm-version: 13.0
// assembly-output: emit-asm
// compile-flags: --target bpfel-unknown-none -C target_feature=+alu32
// needs-llvm-components: bpf

#![feature(no_core, lang_items, rustc_attrs, repr_simd)]
#![crate_type = "rlib"]
#![no_core]
#![allow(asm_sub_register, non_camel_case_types)]

#[rustc_builtin_macro]
macro_rules! asm {
    () => {};
}
#[rustc_builtin_macro]
macro_rules! concat {
    () => {};
}
#[rustc_builtin_macro]
macro_rules! stringify {
    () => {};
}

#[lang = "sized"]
trait Sized {}
#[lang = "copy"]
trait Copy {}

type ptr = *const u64;

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
            asm!("{} = {}", out($class) y, in($class) x);
            y
        }
    };
}

macro_rules! check_reg {
    ($func:ident $ty:ident $reg:tt) => {
        #[no_mangle]
        pub unsafe fn $func(x: $ty) -> $ty {
            let y;
            asm!(concat!($reg, " = ", $reg), lateout($reg) y, in($reg) x);
            y
        }
    };
}

extern "C" {
    fn extern_func();
}

// CHECK-LABEL: sym_fn
// CHECK: #APP
// CHECK: call extern_func
// CHECK: #NO_APP
#[no_mangle]
pub unsafe fn sym_fn() {
    asm!("call {}", sym extern_func);
}

// CHECK-LABEL: reg_i8:
// CHECK: #APP
// CHECK: r{{[0-9]+}} = r{{[0-9]+}}
// CHECK: #NO_APP
check!(reg_i8 i8 reg);

// CHECK-LABEL: reg_i16:
// CHECK: #APP
// CHECK: r{{[0-9]+}} = r{{[0-9]+}}
// CHECK: #NO_APP
check!(reg_i16 i16 reg);

// CHECK-LABEL: reg_i32:
// CHECK: #APP
// CHECK: r{{[0-9]+}} = r{{[0-9]+}}
// CHECK: #NO_APP
check!(reg_i32 i32 reg);

// CHECK-LABEL: reg_i64:
// CHECK: #APP
// CHECK: r{{[0-9]+}} = r{{[0-9]+}}
// CHECK: #NO_APP
check!(reg_i64 i64 reg);

// CHECK-LABEL: wreg_i8:
// CHECK: #APP
// CHECK: w{{[0-9]+}} = w{{[0-9]+}}
// CHECK: #NO_APP
check!(wreg_i8 i8 wreg);

// CHECK-LABEL: wreg_i16:
// CHECK: #APP
// CHECK: w{{[0-9]+}} = w{{[0-9]+}}
// CHECK: #NO_APP
check!(wreg_i16 i16 wreg);

// CHECK-LABEL: wreg_i32:
// CHECK: #APP
// CHECK: w{{[0-9]+}} = w{{[0-9]+}}
// CHECK: #NO_APP
check!(wreg_i32 i32 wreg);

// CHECK-LABEL: r0_i8:
// CHECK: #APP
// CHECK: r0 = r0
// CHECK: #NO_APP
check_reg!(r0_i8 i8 "r0");

// CHECK-LABEL: r0_i16:
// CHECK: #APP
// CHECK: r0 = r0
// CHECK: #NO_APP
check_reg!(r0_i16 i16 "r0");

// CHECK-LABEL: r0_i32:
// CHECK: #APP
// CHECK: r0 = r0
// CHECK: #NO_APP
check_reg!(r0_i32 i32 "r0");

// CHECK-LABEL: r0_i64:
// CHECK: #APP
// CHECK: r0 = r0
// CHECK: #NO_APP
check_reg!(r0_i64 i64 "r0");

// CHECK-LABEL: w0_i8:
// CHECK: #APP
// CHECK: w0 = w0
// CHECK: #NO_APP
check_reg!(w0_i8 i8 "w0");

// CHECK-LABEL: w0_i16:
// CHECK: #APP
// CHECK: w0 = w0
// CHECK: #NO_APP
check_reg!(w0_i16 i16 "w0");

// CHECK-LABEL: w0_i32:
// CHECK: #APP
// CHECK: w0 = w0
// CHECK: #NO_APP
check_reg!(w0_i32 i32 "w0");
