// revisions: s390x
// assembly-output: emit-asm
//[s390x] compile-flags: --target s390x-unknown-linux-gnu
//[s390x] needs-llvm-components: systemz

#![feature(no_core, lang_items, rustc_attrs, repr_simd, asm_sym, asm_experimental_arch)]
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

macro_rules! check_reg { ($func:ident, $ty:ty, $reg:tt, $mov:literal) => {
    #[no_mangle]
    pub unsafe fn $func(x: $ty) -> $ty {
        dont_merge(stringify!($func));

        let y;
        asm!(concat!($mov, " %", $reg, ", %", $reg), lateout($reg) y, in($reg) x);
        y
    }
};}

// CHECK-LABEL: sym_fn_32:
// CHECK: #APP
// CHECK: brasl %r14, extern_func
// CHECK: #NO_APP
#[cfg(s390x)]
#[no_mangle]
pub unsafe fn sym_fn_32() {
    asm!("brasl %r14, {}", sym extern_func);
}

// CHECK-LABEL: sym_static:
// CHECK: #APP
// CHECK: brasl %r14, extern_static
// CHECK: #NO_APP
#[no_mangle]
pub unsafe fn sym_static() {
    asm!("brasl %r14, {}", sym extern_static);
}

// CHECK-LABEL: reg_i8:
// CHECK: #APP
// CHECK: lgr %r{{[0-9]+}}, %r{{[0-9]+}}
// CHECK: #NO_APP
check!(reg_i8, i8, reg, "lgr");

// CHECK-LABEL: reg_i16:
// CHECK: #APP
// CHECK: lgr %r{{[0-9]+}}, %r{{[0-9]+}}
// CHECK: #NO_APP
check!(reg_i16, i16, reg, "lgr");

// CHECK-LABEL: reg_i32:
// CHECK: #APP
// CHECK: lgr %r{{[0-9]+}}, %r{{[0-9]+}}
// CHECK: #NO_APP
check!(reg_i32, i32, reg, "lgr");

// CHECK-LABEL: reg_i64:
// CHECK: #APP
// CHECK: lgr %r{{[0-9]+}}, %r{{[0-9]+}}
// CHECK: #NO_APP
check!(reg_i64, i64, reg, "lgr");

// CHECK-LABEL: reg_f32:
// CHECK: #APP
// CHECK: ler %f{{[0-9]+}}, %f{{[0-9]+}}
// CHECK: #NO_APP
check!(reg_f32, f32, freg, "ler");

// CHECK-LABEL: reg_f64:
// CHECK: #APP
// CHECK: ldr %f{{[0-9]+}}, %f{{[0-9]+}}
// CHECK: #NO_APP
check!(reg_f64, f64, freg, "ldr");

// CHECK-LABEL: reg_ptr:
// CHECK: #APP
// CHECK: lgr %r{{[0-9]+}}, %r{{[0-9]+}}
// CHECK: #NO_APP
check!(reg_ptr, ptr, reg, "lgr");

// CHECK-LABEL: r0_i8:
// CHECK: #APP
// CHECK: lr %r0, %r0
// CHECK: #NO_APP
check_reg!(r0_i8, i8, "r0", "lr");

// CHECK-LABEL: r0_i16:
// CHECK: #APP
// CHECK: lr %r0, %r0
// CHECK: #NO_APP
check_reg!(r0_i16, i16, "r0", "lr");

// CHECK-LABEL: r0_i32:
// CHECK: #APP
// CHECK: lr %r0, %r0
// CHECK: #NO_APP
check_reg!(r0_i32, i32, "r0", "lr");

// CHECK-LABEL: r0_i64:
// CHECK: #APP
// CHECK: lr %r0, %r0
// CHECK: #NO_APP
check_reg!(r0_i64, i64, "r0", "lr");

// CHECK-LABEL: f0_f32:
// CHECK: #APP
// CHECK: ler %f0, %f0
// CHECK: #NO_APP
check_reg!(f0_f32, f32, "f0", "ler");

// CHECK-LABEL: f0_f64:
// CHECK: #APP
// CHECK: ldr %f0, %f0
// CHECK: #NO_APP
check_reg!(f0_f64, f64, "f0", "ldr");
