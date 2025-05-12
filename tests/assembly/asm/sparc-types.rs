//@ add-core-stubs
//@ revisions: sparc sparcv8plus sparc64
//@ assembly-output: emit-asm
//@[sparc] compile-flags: --target sparc-unknown-none-elf
//@[sparc] needs-llvm-components: sparc
//@[sparcv8plus] compile-flags: --target sparc-unknown-linux-gnu
//@[sparcv8plus] needs-llvm-components: sparc
//@[sparc64] compile-flags: --target sparc64-unknown-linux-gnu
//@[sparc64] needs-llvm-components: sparc
//@ compile-flags: -Zmerge-functions=disabled

#![feature(no_core, asm_experimental_arch)]
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

macro_rules! check { ($func:ident, $ty:ty, $class:ident, $mov:literal) => {
    #[no_mangle]
    pub unsafe fn $func(x: $ty) -> $ty {
        let y;
        asm!(concat!($mov," {}, {}"), in($class) x, out($class) y);
        y
    }
};}

macro_rules! check_reg { ($func:ident, $ty:ty, $reg:tt, $mov:literal) => {
    #[no_mangle]
    pub unsafe fn $func(x: $ty) -> $ty {
        let y;
        asm!(concat!($mov, " %", $reg, ", %", $reg), in($reg) x, lateout($reg) y);
        y
    }
};}

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
