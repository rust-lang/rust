//@ add-minicore
//@ revisions: sparc sparcv8plus sparc64
//@ assembly-output: emit-asm
//@[sparc] compile-flags: --target sparc-unknown-none-elf
//@[sparc] needs-llvm-components: sparc
//@[sparcv8plus] compile-flags: --target sparc-unknown-linux-gnu --cfg v9
//@[sparcv8plus] needs-llvm-components: sparc
//@[sparcv8plus] filecheck-flags: --check-prefix v9
//@[sparc64] compile-flags: --target sparc64-unknown-linux-gnu --cfg v9
//@[sparc64] needs-llvm-components: sparc
//@[sparc64] filecheck-flags: --check-prefix v9
//@ compile-flags: -Zmerge-functions=disabled -Copt-level=3
//@ compile-flags: --check-cfg=cfg(v9)
//@ min-llvm-version: 22

#![deny(unexpected_cfgs)]
#![feature(no_core, asm_experimental_arch, f128)]
#![crate_type = "rlib"]
#![no_core]
#![allow(asm_sub_register, non_camel_case_types)]

#[cfg_attr(v9, cfg(not(target_feature = "v9")))]
#[cfg_attr(not(v9), cfg(target_feature = "v9"))]
compile_error!("v9 cfg mismatch");

extern crate minicore;
use minicore::*;

type ptr = *const i32;

extern "C" {
    fn extern_func();
    static extern_static: u8;
}

macro_rules! check { ($func:ident, $ty:ty, $class:ident, $mov:literal) => {
    #[unsafe(no_mangle)]
    pub unsafe fn $func(x: $ty) -> $ty {
        let y;
        asm!(concat!($mov," {}, {}"), in($class) x, out($class) y);
        y
    }
};}

macro_rules! check_reg {
    ($func:ident, $ty:ty, $reg:tt, $mov:literal) => {
        #[unsafe(no_mangle)]
        pub unsafe fn $func(x: $ty) -> $ty {
            let y;
            asm!(concat!($mov, " %", $reg, ", %", $reg), in($reg) x, lateout($reg) y);
            y
        }
    };
    ($func:ident, $ty:ty, $reg:tt, $asm_reg:tt, $mov:literal) => {
        #[unsafe(no_mangle)]
        pub unsafe fn $func(x: $ty) -> $ty {
            let y;
            asm!(concat!($mov, " %", $asm_reg, ", %", $asm_reg), in($reg) x, lateout($reg) y);
            y
        }
    };
}

// CHECK-LABEL: sym_fn_32:
// CHECK: !APP
// CHECK-NEXT: call extern_func
// CHECK-NEXT: !NO_APP
#[no_mangle]
pub unsafe fn sym_fn_32() {
    asm!("call {}", sym extern_func);
}

// CHECK-LABEL: sym_static:
// CHECK: !APP
// CHECK-NEXT: call extern_static
// CHECK-NEXT: !NO_APP
#[no_mangle]
pub unsafe fn sym_static() {
    asm!("call {}", sym extern_static);
}

// CHECK-LABEL: reg_i8:
// CHECK: !APP
// CHECK-NEXT: mov %{{[goli]}}{{[0-9]+}}, %{{[goli]}}{{[0-9]+}}
// CHECK-NEXT: !NO_APP
check!(reg_i8, i8, reg, "mov");

// CHECK-LABEL: reg_i16:
// CHECK: !APP
// CHECK-NEXT: mov %{{[goli]}}{{[0-9]+}}, %{{[goli]}}{{[0-9]+}}
// CHECK-NEXT: !NO_APP
check!(reg_i16, i16, reg, "mov");

// CHECK-LABEL: reg_i32:
// CHECK: !APP
// CHECK-NEXT: mov %{{[goli]}}{{[0-9]+}}, %{{[goli]}}{{[0-9]+}}
// CHECK-NEXT: !NO_APP
check!(reg_i32, i32, reg, "mov");

// FIXME: should be allowed for sparcv8plus but not yet supported in LLVM
// sparc64-LABEL: reg_i64:
// sparc64: !APP
// sparc64-NEXT: mov %{{[goli]}}{{[0-9]+}}, %{{[goli]}}{{[0-9]+}}
// sparc64-NEXT: !NO_APP
#[cfg(sparc64)]
check!(reg_i64, i64, reg, "mov");

// CHECK-LABEL: reg_ptr:
// CHECK: !APP
// CHECK-NEXT: mov %{{[goli]}}{{[0-9]+}}, %{{[goli]}}{{[0-9]+}}
// CHECK-NEXT: !NO_APP
check!(reg_ptr, ptr, reg, "mov");

// CHECK-LABEL: o0_i8:
// CHECK: !APP
// CHECK-NEXT: mov %o0, %o0
// CHECK-NEXT: !NO_APP
check_reg!(o0_i8, i8, "o0", "mov");

// CHECK-LABEL: o0_i16:
// CHECK: !APP
// CHECK-NEXT: mov %o0, %o0
// CHECK-NEXT: !NO_APP
check_reg!(o0_i16, i16, "o0", "mov");

// CHECK-LABEL: o0_i32:
// CHECK: !APP
// CHECK-NEXT: mov %o0, %o0
// CHECK-NEXT: !NO_APP
check_reg!(o0_i32, i32, "o0", "mov");

// FIXME: should be allowed for sparcv8plus but not yet supported in LLVM
// sparc64-LABEL: o0_i64:
// sparc64: !APP
// sparc64-NEXT: mov %o0, %o0
// sparc64-NEXT: !NO_APP
#[cfg(sparc64)]
check_reg!(o0_i64, i64, "o0", "mov");

// CHECK-LABEL: r9_i8:
// CHECK: !APP
// CHECK-NEXT: mov %o1, %o1
// CHECK-NEXT: !NO_APP
check_reg!(r9_i8, i8, "r9", "mov");

// CHECK-LABEL: r9_i16:
// CHECK: !APP
// CHECK-NEXT: mov %o1, %o1
// CHECK-NEXT: !NO_APP
check_reg!(r9_i16, i16, "r9", "mov");

// CHECK-LABEL: r9_i32:
// CHECK: !APP
// CHECK-NEXT: mov %o1, %o1
// CHECK-NEXT: !NO_APP
check_reg!(r9_i32, i32, "r9", "mov");

// FIXME: should be allowed for sparcv8plus but not yet supported in LLVM
// sparc64-LABEL: r9_i64:
// sparc64: !APP
// sparc64-NEXT: mov %o1, %o1
// sparc64-NEXT: !NO_APP
#[cfg(sparc64)]
check_reg!(r9_i64, i64, "r9", "mov");

// CHECK-LABEL: freg_f32:
// CHECK: !APP
// CHECK-NEXT: fmovs %f{{[0-9]+}}, %f{{[0-9]+}}
// CHECK-NEXT: !NO_APP
check!(freg_f32, f32, freg, "fmovs");

// CHECK-LABEL: dreg_f64:
// CHECK: !APP
// CHECK-NEXT: fmovs %f{{[0-9]+}}, %f{{[0-9]+}}
// CHECK-NEXT: !NO_APP
check!(dreg_f64, f64, dreg, "fmovs");

// CHECK-LABEL: qreg_f128:
// CHECK: !APP
// CHECK-NEXT: fmovs %f{{[0-9]+}}, %f{{[0-9]+}}
// CHECK-NEXT: !NO_APP
check!(qreg_f128, f128, qreg, "fmovs");

// CHECK-LABEL: f0_f32:
// CHECK: !APP
// CHECK-NEXT: fmovs %f0, %f0
// CHECK-NEXT: !NO_APP
check_reg!(f0_f32, f32, "f0", "fmovs");

// CHECK-LABEL: d0_f64:
// CHECK: !APP
// CHECK-NEXT: fmovs %f0, %f0
// CHECK-NEXT: !NO_APP
check_reg!(d0_f64, f64, "d0", "f0", "fmovs");

// CHECK-LABEL: q0_f128:
// CHECK: !APP
// CHECK-NEXT: fmovs %f0, %f0
// CHECK-NEXT: !NO_APP
check_reg!(q0_f128, f128, "q0", "f0", "fmovs");

// v9-LABEL: d62_f64:
// v9: !APP
// v9-NEXT: fmovd %f62, %f62
// v9-NEXT: !NO_APP
#[cfg(v9)]
check_reg!(d62_f64, f64, "d62", "f62", "fmovd");

// v9-LABEL: q60_f128:
// v9: !APP
// v9-NEXT: fmovd %f60, %f60
// v9-NEXT: !NO_APP
#[cfg(v9)]
check_reg!(q60_f128, f128, "q60", "f60", "fmovd");
