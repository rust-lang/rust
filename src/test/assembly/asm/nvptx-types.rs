// no-system-llvm
// assembly-output: emit-asm
// compile-flags: --target --nvptx64-nvidia-cuda
// only-nvptx64
// ignore-nvptx64

#![feature(no_core, lang_items, rustc_attrs)]
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

type ptr = *mut u8;

impl Copy for i8 {}
impl Copy for i16 {}
impl Copy for i32 {}
impl Copy for f32 {}
impl Copy for i64 {}
impl Copy for f64 {}
impl Copy for ptr {}

#[no_mangle]
fn extern_func();

// CHECK-LABEL: sym_fn
// CHECK: #APP
// CHECK call extern_func;
// CHECK: #NO_APP
#[no_mangle]
pub unsafe fn sym_fn() {
    asm!("call {}", sym extern_func);
}

macro_rules! check {
    ($func:ident $ty:ident, $class:ident $mov:literal) => {
        #[no_mangle]
        pub unsafe fn $func(x: $ty) -> $ty {
            // Hack to avoid function merging
            extern "Rust" {
                fn dont_merge(s: &str);
            }
            dont_merge(stringify!($func));

            let y;
            asm!(concat!($mov, " {}, {};"), out($class) y, in($class) x);
            y
        }
    };
}

// CHECK-LABEL: reg_i8
// CHECK: #APP
// CHECK: mov.i16 {{[a-z0-9]+}}, {{[a-z0-9]+}};
// CHECK: #NO_APP
check!(reg_i8 i8 reg16 "mov.i16");

// CHECK-LABEL: reg_i16
// CHECK: #APP
// CHECK: mov.i16 {{[a-z0-9]+}}, {{[a-z0-9]+}};
// CHECK: #NO_APP
check!(reg_i16 i16 reg16 "mov.i16");

// CHECK-LABEL: reg_i32
// CHECK: #APP
// CHECK: mov.i32 {{[a-z0-9]+}}, {{[a-z0-9]+}};
// CHECK: #NO_APP
check!(reg_i32 i32 reg32 "mov.i32");

// CHECK-LABEL: reg_f32
// CHECK: #APP
// CHECK: mov.f32 {{[a-z0-9]+}}, {{[a-z0-9]+}};
// CHECK: #NO_APP
check!(reg_f32 f32 freg32 "mov.f32");

// CHECK-LABEL: reg_i54
// CHECK: #APP
// CHECK: mov.i64 {{[a-z0-9]+}}, {{[a-z0-9]+}};
// CHECK: #NO_APP
check!(reg_i64 i64 reg64 "mov.i64");

// CHECK-LABEL: reg_f64
// CHECK: #APP
// CHECK: mov.f64 {{[a-z0-9]+}}, {{[a-z0-9]+}};
// CHECK: #NO_APP
check!(reg_f64 f64 freg64 "mov.f64");

// CHECK-LABEL: reg_ptr
// CHECK: #APP
// CHECK: mov.i64 {{[a-z0-9]+}}, {{[a-z0-9]+}};
// CHECK: #NO_APP
check!(reg_ptr ptr reg64 "mov.i64");
