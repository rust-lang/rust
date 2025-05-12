//@ add-core-stubs
//@ assembly-output: emit-asm
//@ compile-flags: --target m68k-unknown-linux-gnu
//@ needs-llvm-components: m68k

#![feature(no_core, asm_experimental_arch)]
#![crate_type = "rlib"]
#![no_core]
#![allow(non_camel_case_types)]

extern crate minicore;
use minicore::*;

type ptr = *const u64;

macro_rules! check {
    ($func:ident $ty:ident $class:ident $mov:literal) => {
        #[no_mangle]
        pub unsafe fn $func(x: $ty) -> $ty {
            let y;
            asm!(concat!($mov, " {}, {};"), out($class) y, in($class) x);
            y
        }
    };
}

// CHECK-LABEL: reg_data_i8:
// CHECK: ;APP
// CHECK: move.b %d{{[0-9]}}, %d{{[0-9]}}
// CHECK: ;NO_APP
check!(reg_data_i8 i8 reg_data "move.b");

// CHECK-LABEL: reg_data_i16:
// CHECK: ;APP
// CHECK: move.w %d{{[0-9]}}, %d{{[0-9]}}
// CHECK: ;NO_APP
check!(reg_data_i16 i16 reg_data "move.w");

// CHECK-LABEL: reg_data_i32:
// CHECK: ;APP
// CHECK: move.l %d{{[0-9]}}, %d{{[0-9]}}
// CHECK: ;NO_APP
check!(reg_data_i32 i32 reg_data "move.l");

// CHECK-LABEL: reg_addr_i16:
// CHECK: ;APP
// CHECK: move.w %a{{[0-9]}}, %a{{[0-9]}}
// CHECK: ;NO_APP
check!(reg_addr_i16 i16 reg_addr "move.w");

// CHECK-LABEL: reg_addr_i32:
// CHECK: ;APP
// CHECK: move.l %a{{[0-9]}}, %a{{[0-9]}}
// CHECK: ;NO_APP
check!(reg_addr_i32 i32 reg_addr "move.l");

// CHECK-LABEL: reg_i16:
// CHECK: ;APP
// CHECK: move.w %{{[da][0-9]}}, %{{[da][0-9]}}
// CHECK: ;NO_APP
check!(reg_i16 i16 reg "move.w");

// CHECK-LABEL: reg_i32:
// CHECK: ;APP
// CHECK: move.l %{{[da][0-9]}}, %{{[da][0-9]}}
// CHECK: ;NO_APP
check!(reg_i32 i32 reg "move.l");
