// min-llvm-version: 12.0.1
// revisions: powerpc powerpc64
// assembly-output: emit-asm
//[powerpc] compile-flags: --target powerpc-unknown-linux-gnu
//[powerpc] needs-llvm-components: powerpc
//[powerpc64] compile-flags: --target powerpc64-unknown-linux-gnu
//[powerpc64] needs-llvm-components: powerpc

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

type ptr = *const i32;

impl Copy for i8 {}
impl Copy for u8 {}
impl Copy for i16 {}
impl Copy for i32 {}
impl Copy for i64 {}
impl Copy for f32 {}
impl Copy for f64 {}
impl Copy for ptr {}
extern "C" {
    fn extern_func();
    static extern_static: u8;
}

// Hack to avoid function merging
extern "Rust" {
    fn dont_merge(s: &str);
}

macro_rules! check { ($func:ident, $ty:ty, $class:ident, $mov:literal) => {
    #[no_mangle]
    pub unsafe fn $func(x: $ty) -> $ty {
        dont_merge(stringify!($func));

        let y;
        asm!(concat!($mov," {}, {}"), out($class) y, in($class) x);
        y
    }
};}

macro_rules! check_reg { ($func:ident, $ty:ty, $rego:tt, $regc:tt, $mov:literal) => {
    #[no_mangle]
    pub unsafe fn $func(x: $ty) -> $ty {
        dont_merge(stringify!($func));

        let y;
        asm!(concat!($mov, " ", $rego, ", ", $rego), lateout($regc) y, in($regc) x);
        y
    }
};}

// CHECK-LABEL: reg_i8:
// CHECK: #APP
// CHECK: mr {{[0-9]+}}, {{[0-9]+}}
// CHECK: #NO_APP
check!(reg_i8, i8, reg, "mr");

// CHECK-LABEL: reg_i16:
// CHECK: #APP
// CHECK: mr {{[0-9]+}}, {{[0-9]+}}
// CHECK: #NO_APP
check!(reg_i16, i16, reg, "mr");

// CHECK-LABEL: reg_i32:
// CHECK: #APP
// CHECK: mr {{[0-9]+}}, {{[0-9]+}}
// CHECK: #NO_APP
check!(reg_i32, i32, reg, "mr");

// powerpc64-LABEL: reg_i64:
// powerpc64: #APP
// powerpc64: mr {{[0-9]+}}, {{[0-9]+}}
// powerpc64: #NO_APP
#[cfg(powerpc64)]
check!(reg_i64, i64, reg, "mr");

// CHECK-LABEL: reg_i8_nz:
// CHECK: #APP
// CHECK: mr {{[0-9]+}}, {{[0-9]+}}
// CHECK: #NO_APP
check!(reg_i8_nz, i8, reg_nonzero, "mr");

// CHECK-LABEL: reg_i16_nz:
// CHECK: #APP
// CHECK: mr {{[0-9]+}}, {{[0-9]+}}
// CHECK: #NO_APP
check!(reg_i16_nz, i16, reg_nonzero, "mr");

// CHECK-LABEL: reg_i32_nz:
// CHECK: #APP
// CHECK: mr {{[0-9]+}}, {{[0-9]+}}
// CHECK: #NO_APP
check!(reg_i32_nz, i32, reg_nonzero, "mr");

// powerpc64-LABEL: reg_i64_nz:
// powerpc64: #APP
// powerpc64: mr {{[0-9]+}}, {{[0-9]+}}
// powerpc64: #NO_APP
#[cfg(powerpc64)]
check!(reg_i64_nz, i64, reg_nonzero, "mr");

// CHECK-LABEL: reg_f32:
// CHECK: #APP
// CHECK: fmr {{[0-9]+}}, {{[0-9]+}}
// CHECK: #NO_APP
check!(reg_f32, f32, freg, "fmr");

// CHECK-LABEL: reg_f64:
// CHECK: #APP
// CHECK: fmr {{[0-9]+}}, {{[0-9]+}}
// CHECK: #NO_APP
check!(reg_f64, f64, freg, "fmr");

// CHECK-LABEL: reg_i8_r0:
// CHECK: #APP
// CHECK: mr 0, 0
// CHECK: #NO_APP
check_reg!(reg_i8_r0, i8, "0", "0", "mr");

// CHECK-LABEL: reg_i16_r0:
// CHECK: #APP
// CHECK: mr 0, 0
// CHECK: #NO_APP
check_reg!(reg_i16_r0, i16, "0", "0", "mr");

// CHECK-LABEL: reg_i32_r0:
// CHECK: #APP
// CHECK: mr 0, 0
// CHECK: #NO_APP
check_reg!(reg_i32_r0, i32, "0", "0", "mr");

// powerpc64-LABEL: reg_i64_r0:
// powerpc64: #APP
// powerpc64: mr 0, 0
// powerpc64: #NO_APP
#[cfg(powerpc64)]
check_reg!(reg_i64_r0, i64, "0", "0", "mr");

// CHECK-LABEL: reg_i8_r18:
// CHECK: #APP
// CHECK: mr 18, 18
// CHECK: #NO_APP
check_reg!(reg_i8_r18, i8, "18", "18", "mr");

// CHECK-LABEL: reg_i16_r18:
// CHECK: #APP
// CHECK: mr 18, 18
// CHECK: #NO_APP
check_reg!(reg_i16_r18, i16, "18", "18", "mr");

// CHECK-LABEL: reg_i32_r18:
// CHECK: #APP
// CHECK: mr 18, 18
// CHECK: #NO_APP
check_reg!(reg_i32_r18, i32, "18", "18", "mr");

// powerpc64-LABEL: reg_i64_r18:
// powerpc64: #APP
// powerpc64: mr 18, 18
// powerpc64: #NO_APP
#[cfg(powerpc64)]
check_reg!(reg_i64_r18, i64, "18", "18", "mr");

// CHECK-LABEL: reg_f32_f0:
// CHECK: #APP
// CHECK: fmr 0, 0
// CHECK: #NO_APP
check_reg!(reg_f32_f0, f32, "0", "f0", "fmr");

// CHECK-LABEL: reg_f64_f0:
// CHECK: #APP
// CHECK: fmr 0, 0
// CHECK: #NO_APP
check_reg!(reg_f64_f0, f64, "0", "f0", "fmr");

// CHECK-LABEL: reg_f32_f18:
// CHECK: #APP
// CHECK: fmr 18, 18
// CHECK: #NO_APP
check_reg!(reg_f32_f18, f32, "18", "f18", "fmr");

// CHECK-LABEL: reg_f64_f18:
// CHECK: #APP
// CHECK: fmr 18, 18
// CHECK: #NO_APP
check_reg!(reg_f64_f18, f64, "18", "f18", "fmr");
