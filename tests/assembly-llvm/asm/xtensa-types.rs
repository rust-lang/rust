//@ add-minicore
//@ assembly-output: emit-asm
//@ compile-flags: --target xtensa-esp32-none-elf -Zmerge-functions=disabled
//@ min-llvm-version: 22
//@ needs-llvm-components: xtensa

#![feature(no_core, lang_items, rustc_attrs, asm_experimental_arch)]
#![crate_type = "rlib"]
#![no_core]
#![allow(asm_sub_register, non_camel_case_types)]

extern crate minicore;
use minicore::simd::*;
use minicore::*;

type ptr = *mut u8;

extern "C" {
    fn extern_func();
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

macro_rules! check_explicit_reg {
    ($func:ident $ty:ident $reg:tt $mov:literal) => {
        #[no_mangle]
        pub unsafe fn $func(x: $ty) -> $ty {
            let y;
            asm!(concat!($mov, " ", $reg, ", ", $reg), lateout($reg) y, in($reg) x);
            y
        }
    };
}

// CHECK-LABEL: a2_i32:
// CHECK: #APP
// CHECK: or a2, a2, a2
// CHECK: #NO_APP
check_explicit_reg!(a2_i32 i32 "a2" "mov");

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

// CHECK-LABEL: a14_i32:
// CHECK: #APP
// CHECK: or a14, a14, a14
// CHECK: #NO_APP
check_explicit_reg!(a14_i32 i32 "a14" "mov");

// a15 is the frame pointer under CALL0, but usable under the windowed ABI
// (this test target).
// CHECK-LABEL: a15_i32:
// CHECK: #APP
// CHECK: or a15, a15, a15
// CHECK: #NO_APP
check_explicit_reg!(a15_i32 i32 "a15" "mov");

// CHECK-LABEL: f0_f32:
// CHECK: #APP
// CHECK: mov.s f0, f0
// CHECK: #NO_APP
check_explicit_reg!(f0_f32 f32 "f0" "mov.s");

// CHECK-LABEL: f7_f32:
// CHECK: #APP
// CHECK: mov.s f7, f7
// CHECK: #NO_APP
check_explicit_reg!(f7_f32 f32 "f7" "mov.s");

// CHECK-LABEL: f15_f32:
// CHECK: #APP
// CHECK: mov.s f15, f15
// CHECK: #NO_APP
check_explicit_reg!(f15_f32 f32 "f15" "mov.s");

// Special/Boolean registers are clobber-only.
macro_rules! check_clobber {
    ($func:ident $reg:tt $insn:literal) => {
        #[no_mangle]
        pub unsafe fn $func(x: i32) {
            asm!($insn, in(reg) x, out($reg) _);
        }
    };
}

// CHECK-LABEL: sar_clobber:
// CHECK: #APP
// CHECK: wsr a{{[0-9]+}}, sar
// CHECK: #NO_APP
check_clobber!(sar_clobber "sar" "wsr {0}, sar");

// CHECK-LABEL: scompare1_clobber:
// CHECK: #APP
// CHECK: wsr a{{[0-9]+}}, scompare1
// CHECK: #NO_APP
check_clobber!(scompare1_clobber "scompare1" "wsr {0}, scompare1");

// CHECK-LABEL: lbeg_clobber:
// CHECK: #APP
// CHECK: wsr a{{[0-9]+}}, lbeg
// CHECK: #NO_APP
check_clobber!(lbeg_clobber "lbeg" "wsr {0}, lbeg");

// CHECK-LABEL: lend_clobber:
// CHECK: #APP
// CHECK: wsr a{{[0-9]+}}, lend
// CHECK: #NO_APP
check_clobber!(lend_clobber "lend" "wsr {0}, lend");

// CHECK-LABEL: lcount_clobber:
// CHECK: #APP
// CHECK: wsr a{{[0-9]+}}, lcount
// CHECK: #NO_APP
check_clobber!(lcount_clobber "lcount" "wsr {0}, lcount");

// CHECK-LABEL: acclo_clobber:
// CHECK: #APP
// CHECK: wsr a{{[0-9]+}}, acclo
// CHECK: #NO_APP
check_clobber!(acclo_clobber "acclo" "wsr {0}, acclo");

// CHECK-LABEL: acchi_clobber:
// CHECK: #APP
// CHECK: wsr a{{[0-9]+}}, acchi
// CHECK: #NO_APP
check_clobber!(acchi_clobber "acchi" "wsr {0}, acchi");

// CHECK-LABEL: m0_clobber:
// CHECK: #APP
// CHECK: wsr a{{[0-9]+}}, m0
// CHECK: #NO_APP
check_clobber!(m0_clobber "m0" "wsr {0}, m0");

// CHECK-LABEL: m3_clobber:
// CHECK: #APP
// CHECK: wsr a{{[0-9]+}}, m3
// CHECK: #NO_APP
check_clobber!(m3_clobber "m3" "wsr {0}, m3");

// Boolean registers are bits of BR; write them via `andb`.
macro_rules! check_breg_clobber {
    ($func:ident $reg:tt) => {
        #[no_mangle]
        pub unsafe fn $func() {
            asm!(concat!("andb ", $reg, ", ", $reg, ", ", $reg), out($reg) _);
        }
    };
}

// CHECK-LABEL: b0_clobber:
// CHECK: #APP
// CHECK: andb b0, b0, b0
// CHECK: #NO_APP
check_breg_clobber!(b0_clobber "b0");

// CHECK-LABEL: b7_clobber:
// CHECK: #APP
// CHECK: andb b7, b7, b7
// CHECK: #NO_APP
check_breg_clobber!(b7_clobber "b7");

// CHECK-LABEL: b15_clobber:
// CHECK: #APP
// CHECK: andb b15, b15, b15
// CHECK: #NO_APP
check_breg_clobber!(b15_clobber "b15");
