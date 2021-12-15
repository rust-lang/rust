// min-llvm-version: 13.0
// assembly-output: emit-asm
// compile-flags: --target avr-unknown-gnu-atmega328
// needs-llvm-components: avr

#![feature(no_core, lang_items, rustc_attrs, asm_experimental_arch)]
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

type ptr = *const u64;

impl Copy for i8 {}
impl Copy for i16 {}
impl Copy for i32 {}
impl Copy for i64 {}
impl Copy for ptr {}

macro_rules! check {
    ($func:ident $hi:literal $lo:literal $reg:tt) => {
        #[no_mangle]
        unsafe fn $func() -> i16 {
            let y;
            asm!(concat!("mov {0:", $hi, "}, {0:", $lo, "}"), out($reg) y);
            y
        }
    };
}

// CHECK-LABEL: reg_pair_modifiers:
// CHECK: ;APP
// CHECK: mov r{{[1-9]?[13579]}}, r{{[1-9]?[24680]}}
// CHECK: ;NO_APP
check!(reg_pair_modifiers "h" "l" reg_pair);

// CHECK-LABEL: reg_iw_modifiers:
// CHECK: ;APP
// CHECK: mov r{{[1-9]?[13579]}}, r{{[1-9]?[24680]}}
// CHECK: ;NO_APP
check!(reg_iw_modifiers "h" "l" reg_iw);

// CHECK-LABEL: reg_ptr_modifiers:
// CHECK: ;APP
// CHECK: mov r{{[1-9]?[13579]}}, r{{[1-9]?[24680]}}
// CHECK: ;NO_APP
check!(reg_ptr_modifiers "h" "l" reg_ptr);
