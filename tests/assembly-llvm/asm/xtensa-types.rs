//@ add-minicore
//@ assembly-output: emit-asm
//@ compile-flags: --target xtensa-esp32-none-elf
//@ needs-llvm-components: xtensa

#![feature(no_core, lang_items, rustc_attrs, repr_simd, asm_experimental_arch)]
#![crate_type = "rlib"]
#![no_core]
#![allow(asm_sub_register, non_camel_case_types)]

extern crate minicore;
use minicore::*;

type ptr = *mut u8;

extern "C" {
    fn extern_func();
}

// Hack to avoid function merging
extern "Rust" {
    fn dont_merge(s: &str);
}

// CHECK-LABEL: sym_fn:
// CHECK: #APP
// CHECK: call4 extern_func
// CHECK: #NO_APP
#[no_mangle]
pub unsafe fn sym_fn() {
    asm!("call4 {}", sym extern_func);
}

macro_rules! check_general_reg {
    ($func:ident $ty:ident $class:ident $mov:literal) => {
        #[no_mangle]
        pub unsafe fn $func(x: $ty) -> $ty {
            dont_merge(stringify!($func));

            let y;
            asm!(concat!($mov, " {}, {}"), out($class) y, in($class) x);
            y
        }
    };
}

// CHECK-LABEL: reg_i8:
// CHECK: #APP
// CHECK: or a{{[0-9]+}}, a{{[0-9]+}}, a{{[0-9]+}}
// CHECK: #NO_APP
check_general_reg!(reg_i8 i8 reg "mov");

// CHECK-LABEL: reg_i16:
// CHECK: #APP
// CHECK: or a{{[0-9]+}}, a{{[0-9]+}}, a{{[0-9]+}}
// CHECK: #NO_APP
check_general_reg!(reg_i16 i16 reg "mov");

// CHECK-LABEL: reg_i32:
// CHECK: #APP
// CHECK: or a{{[0-9]+}}, a{{[0-9]+}}, a{{[0-9]+}}
// CHECK: #NO_APP
check_general_reg!(reg_i32 i32 reg "mov");

// CHECK-LABEL: reg_ptr:
// CHECK: #APP
// CHECK: or a{{[0-9]+}}, a{{[0-9]+}}, a{{[0-9]+}}
// CHECK: #NO_APP
check_general_reg!(reg_ptr ptr reg "mov");

// CHECK-LABEL: freg_f32:
// CHECK: #APP
// CHECK: mov.s f{{[0-9]+}}, f{{[0-9]+}}
// CHECK: #NO_APP
check_general_reg!(freg_f32 f32 freg "mov.s");

macro_rules! check_explicit_reg {
    ($func:ident $ty:ident $reg:tt $mov:literal) => {
        #[no_mangle]
        pub unsafe fn $func(x: $ty) -> $ty {
            dont_merge(stringify!($func));

            let y;
            asm!(concat!($mov, " ", $reg, ", ", $reg), lateout($reg) y, in($reg) x);
            y
        }
    };
}

// CHECK-LABEL: a5_i8:
// CHECK: #APP
// CHECK: or a5, a5, a5
// CHECK: #NO_APP
check_explicit_reg!(a5_i8 i8 "a5" "mov");

// CHECK-LABEL: a5_i16:
// CHECK: #APP
// CHECK: or a5, a5, a5
// CHECK: #NO_APP
check_explicit_reg!(a5_i16 i16 "a5" "mov");

// CHECK-LABEL: a5_i32:
// CHECK: #APP
// CHECK: or a5, a5, a5
// CHECK: #NO_APP
check_explicit_reg!(a5_i32 i32 "a5" "mov");

// CHECK-LABEL: a5_ptr:
// CHECK: #APP
// CHECK: or a5, a5, a5
// CHECK: #NO_APP
check_explicit_reg!(a5_ptr ptr "a5" "mov");

// CHECK-LABEL: f0_f32:
// CHECK: #APP
// CHECK: mov.s f0, f0
// CHECK: #NO_APP
check_explicit_reg!(f0_f32 f32 "f0" "mov.s");
