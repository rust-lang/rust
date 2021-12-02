// assembly-output: emit-asm
// compile-flags: --target hexagon-unknown-linux-musl
// needs-llvm-components: hexagon

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
impl Copy for i16 {}
impl Copy for i32 {}
impl Copy for f32 {}
impl Copy for ptr {}
extern "C" {
    fn extern_func();
    static extern_static: u8;
}

macro_rules! check {
    ($func:ident $ty:ident $class:ident) => {
        #[no_mangle]
        pub unsafe fn $func(x: $ty) -> $ty {
            // Hack to avoid function merging
            extern "Rust" {
                fn dont_merge(s: &str);
            }
            dont_merge(stringify!($func));

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
            // Hack to avoid function merging
            extern "Rust" {
                fn dont_merge(s: &str);
            }
            dont_merge(stringify!($func));

            let y;
            asm!(concat!($reg, " = ", $reg), lateout($reg) y, in($reg) x);
            y
        }
    };
}

// CHECK-LABEL: sym_static:
// CHECK: InlineAsm Start
// CHECK: r0 = #extern_static
// CHECK: InlineAsm End
#[no_mangle]
pub unsafe fn sym_static() {
    // Hack to avoid function merging
    extern "Rust" {
        fn dont_merge(s: &str);
    }
    dont_merge(stringify!($func));

    asm!("r0 = #{}", sym extern_static);
}

// CHECK-LABEL: sym_fn:
// CHECK: InlineAsm Start
// CHECK: r0 = #extern_func
// CHECK: InlineAsm End
#[no_mangle]
pub unsafe fn sym_fn() {
    // Hack to avoid function merging
    extern "Rust" {
        fn dont_merge(s: &str);
    }
    dont_merge(stringify!($func));

    asm!("r0 = #{}", sym extern_func);
}

// This is a test for multi-instruction packets,
// which require the escaped braces.
//
// CHECK-LABEL: packet:
// CHECK: InlineAsm Start
// CHECK: {
// CHECK:   r{{[0-9]+}} = r0
// CHECK:   memw(r1) = r{{[0-9]+}}
// CHECK: }
// CHECK: InlineAsm End
#[no_mangle]
pub unsafe fn packet() {
    let val = 1024;
    asm!("{{
        {} = r0
        memw(r1) = {}
    }}", out(reg) _, in(reg) &val);
}

// CHECK-LABEL: reg_ptr:
// CHECK: InlineAsm Start
// CHECK: r{{[0-9]+}} = r{{[0-9]+}}
// CHECK: InlineAsm End
check!(reg_ptr ptr reg);

// CHECK-LABEL: reg_f32:
// CHECK: InlineAsm Start
// CHECK: r{{[0-9]+}} = r{{[0-9]+}}
// CHECK: InlineAsm End
check!(reg_f32 f32 reg);

// CHECK-LABEL: reg_i32:
// CHECK: InlineAsm Start
// CHECK: r{{[0-9]+}} = r{{[0-9]+}}
// CHECK: InlineAsm End
check!(reg_i32 i32 reg);

// CHECK-LABEL: reg_i8:
// CHECK: InlineAsm Start
// CHECK: r{{[0-9]+}} = r{{[0-9]+}}
// CHECK: InlineAsm End
check!(reg_i8 i8 reg);

// CHECK-LABEL: reg_i16:
// CHECK: InlineAsm Start
// CHECK: r{{[0-9]+}} = r{{[0-9]+}}
// CHECK: InlineAsm End
check!(reg_i16 i16 reg);

// CHECK-LABEL: r0_ptr:
// CHECK: InlineAsm Start
// CHECK: r0 = r0
// CHECK: InlineAsm End
check_reg!(r0_ptr ptr "r0");

// CHECK-LABEL: r0_f32:
// CHECK: InlineAsm Start
// CHECK: r0 = r0
// CHECK: InlineAsm End
check_reg!(r0_f32 f32 "r0");

// CHECK-LABEL: r0_i32:
// CHECK: InlineAsm Start
// CHECK: r0 = r0
// CHECK: InlineAsm End
check_reg!(r0_i32 i32 "r0");

// CHECK-LABEL: r0_i8:
// CHECK: InlineAsm Start
// CHECK: r0 = r0
// CHECK: InlineAsm End
check_reg!(r0_i8 i8 "r0");

// CHECK-LABEL: r0_i16:
// CHECK: InlineAsm Start
// CHECK: r0 = r0
// CHECK: InlineAsm End
check_reg!(r0_i16 i16 "r0");
