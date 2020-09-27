// no-system-llvm
// assembly-output: emit-asm
// compile-flags: --target mips-unknown-linux-gnu
// needs-llvm-components: mips

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
impl Copy for f32 {}
impl Copy for ptr {}
extern "C" {
    fn extern_func();
    static extern_static: u8;
}

// Hack to avoid function merging
extern "Rust" {
    fn dont_merge(s: &str);
}

macro_rules! check { ($func:ident, $ty:ty, $class:ident) => {
    #[no_mangle]
    pub unsafe fn $func(x: $ty) -> $ty {
        dont_merge(stringify!($func));

        let y;
        asm!("move {}, {}", out($class) y, in($class) x);
        y
    }
};}

macro_rules! check_reg { ($func:ident, $ty:ty, $reg:tt) => {
    #[no_mangle]
    pub unsafe fn $func(x: $ty) -> $ty {
        dont_merge(stringify!($func));

        let y;
        asm!(concat!("move ", $reg, ", ", $reg), lateout($reg) y, in($reg) x);
        y
    }
};}

// CHECK-LABEL: sym_static:
// CHECK: #APP
// CHECK: lw $3, %got(extern_static)
// CHECK: #NO_APP
#[no_mangle]
pub unsafe fn sym_static() {
    dont_merge(stringify!($func));

    asm!("la $v1, {}", sym extern_static);
}

// CHECK-LABEL: sym_fn:
// CHECK: #APP
// CHECK: lw $3, %got(extern_func)
// CHECK: #NO_APP
#[no_mangle]
pub unsafe fn sym_fn() {
    dont_merge(stringify!($func));

    asm!("la $v1, {}", sym extern_func);
}

// CHECK-LABEL: reg_f32:
// CHECK: #APP
// CHECK: mov.s $f{{[0-9]+}}, $f{{[0-9]+}}
// CHECK: #NO_APP
#[no_mangle]
pub unsafe fn reg_f32(x: f32) -> f32 {
    dont_merge("reg_f32");
    let y;
    asm!("mov.s {}, {}", out(freg) y, in(freg) x);
    y
}

// CHECK-LABEL: f0_f32:
// CHECK: #APP
// CHECK: mov.s $f0, $f0
// CHECK: #NO_APP
#[no_mangle]
pub unsafe fn f0_f32(x: f32) -> f32 {
    dont_merge("f0_f32");
    let y;
    asm!("mov.s $f0, $f0", lateout("$f0") y, in("$f0") x);
    y
}

// CHECK-LABEL: reg_ptr:
// CHECK: #APP
// CHECK: move ${{[0-9]+}}, ${{[0-9]+}}
// CHECK: #NO_APP
check!(reg_ptr, ptr, reg);

// CHECK-LABEL: reg_i32:
// CHECK: #APP
// CHECK: move ${{[0-9]+}}, ${{[0-9]+}}
// CHECK: #NO_APP
check!(reg_i32, i32, reg);

// CHECK-LABEL: reg_f32_soft:
// CHECK: #APP
// CHECK: move ${{[0-9]+}}, ${{[0-9]+}}
// CHECK: #NO_APP
check!(reg_f32_soft, f32, reg);

// CHECK-LABEL: reg_i8:
// CHECK: #APP
// CHECK: move ${{[0-9]+}}, ${{[0-9]+}}
// CHECK: #NO_APP
check!(reg_i8, i8, reg);

// CHECK-LABEL: reg_u8:
// CHECK: #APP
// CHECK: move ${{[0-9]+}}, ${{[0-9]+}}
// CHECK: #NO_APP
check!(reg_u8, u8, reg);

// CHECK-LABEL: reg_i16:
// CHECK: #APP
// CHECK: move ${{[0-9]+}}, ${{[0-9]+}}
// CHECK: #NO_APP
check!(reg_i16, i16, reg);

// CHECK-LABEL: t0_ptr:
// CHECK: #APP
// CHECK: move $8, $8
// CHECK: #NO_APP
check_reg!(t0_ptr, ptr, "$t0");

// CHECK-LABEL: t0_i32:
// CHECK: #APP
// CHECK: move $8, $8
// CHECK: #NO_APP
check_reg!(t0_i32, i32, "$t0");

// CHECK-LABEL: t0_f32:
// CHECK: #APP
// CHECK: move $8, $8
// CHECK: #NO_APP
check_reg!(t0_f32, f32, "$t0");

// CHECK-LABEL: t0_i8:
// CHECK: #APP
// CHECK: move $8, $8
// CHECK: #NO_APP
check_reg!(t0_i8, i8, "$t0");

// CHECK-LABEL: t0_u8:
// CHECK: #APP
// CHECK: move $8, $8
// CHECK: #NO_APP
check_reg!(t0_u8, u8, "$t0");

// CHECK-LABEL: t0_i16:
// CHECK: #APP
// CHECK: move $8, $8
// CHECK: #NO_APP
check_reg!(t0_i16, i16, "$t0");

// CHECK-LABEL: r8_i16:
// CHECK: #APP
// CHECK: move $8, $8
// CHECK: #NO_APP
check_reg!(r8_i16, i16, "$8");
