// min-llvm-version: 10.0.1
// revisions: s390x
// assembly-output: emit-asm
//[s390x] compile-flags: --target s390x-unknown-linux-gnu
//[s390x] needs-llvm-components: systemz

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
    
    pub unsafe fn $func(x: $ty) -> $ty {
        dont_merge(stringify!(func));

        let y;
        asm!(concat!($mov," {}, {}"), out($class) y, in($class) x);
        y
    }
};}

macro_rules! check_reg { ($func:ident, $ty:ty, $reg:tt, $mov:literal) => {
    
    pub unsafe fn $func(x: $ty) -> $ty {
        dont_merge(stringify!(func));

        let y;
        asm!(concat!($mov, " ", $reg, ", ", $reg), lateout($reg) y, in($reg) x);
        y
    }
};}

// systemz-LABEL: sym_fn_32:
// systemz: #APP
// systemz: brasl %r14, extern_func@PLT
// systemz: #NO_APP
#[cfg(s390x)]
pub unsafe fn sym_fn_32() {
    asm!("brasl %r14, {}", sym extern_func);
}

// CHECK-LABEL: reg_i32:
// CHECK: #APP
// CHECK: lgr r{{[0-15]+}}, r{{[0-15]+}}
// CHECK: #NO_APP
check!(reg_i32, i32, reg, "lgr");
