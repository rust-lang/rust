// revisions: mips32 mips64
// assembly-output: emit-asm
//[mips32] compile-flags: --target mips-unknown-linux-gnu
//[mips32] needs-llvm-components: mips
//[mips64] compile-flags: --target mips64-unknown-linux-gnuabi64
//[mips64] needs-llvm-components: mips

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
        asm!(concat!($mov, " ", $reg, ", ", $reg), lateout($reg) y, in($reg) x);
        y
    }
};}

// mips32-LABEL: sym_static_32:
// mips32: #APP
// mips32: lw $3, %got(extern_static)
// mips32: #NO_APP
#[cfg(mips32)]
#[no_mangle]
pub unsafe fn sym_static_32() {
    asm!("lw $v1, {}", sym extern_static);
}

// mips32-LABEL: sym_fn_32:
// mips32: #APP
// mips32: lw $3, %got(extern_func)
// mips32: #NO_APP
#[cfg(mips32)]
#[no_mangle]
pub unsafe fn sym_fn_32() {
    asm!("lw $v1, {}", sym extern_func);
}

// mips64-LABEL: sym_static_64:
// mips64: #APP
// mips64: ld $3, %got_disp(extern_static)
// mips64: #NO_APP
#[cfg(mips64)]
#[no_mangle]
pub unsafe fn sym_static_64() {
    asm!("ld $v1, {}", sym extern_static);
}

// mips64-LABEL: sym_fn_64:
// mips64: #APP
// mips64: ld $3, %got_disp(extern_func)
// mips64: #NO_APP
#[cfg(mips64)]
#[no_mangle]
pub unsafe fn sym_fn_64() {
    asm!("ld $v1, {}", sym extern_func);
}

// CHECK-LABEL: reg_f32:
// CHECK: #APP
// CHECK: mov.s $f{{[0-9]+}}, $f{{[0-9]+}}
// CHECK: #NO_APP
check!(reg_f32, f32, freg, "mov.s");

// CHECK-LABEL: f0_f32:
// CHECK: #APP
// CHECK: mov.s $f0, $f0
// CHECK: #NO_APP
#[no_mangle]
check_reg!(f0_f32, f32, "$f0", "mov.s");

// CHECK-LABEL: reg_f32_64:
// CHECK: #APP
// CHECK: mov.d $f{{[0-9]+}}, $f{{[0-9]+}}
// CHECK: #NO_APP
check!(reg_f32_64, f32, freg, "mov.d");

// CHECK-LABEL: f0_f32_64:
// CHECK: #APP
// CHECK: mov.d $f0, $f0
// CHECK: #NO_APP
#[no_mangle]
check_reg!(f0_f32_64, f32, "$f0", "mov.d");

// CHECK-LABEL: reg_f64:
// CHECK: #APP
// CHECK: mov.d $f{{[0-9]+}}, $f{{[0-9]+}}
// CHECK: #NO_APP
#[no_mangle]
check!(reg_f64, f64, freg, "mov.d");

// CHECK-LABEL: f0_f64:
// CHECK: #APP
// CHECK: mov.d $f0, $f0
// CHECK: #NO_APP
#[no_mangle]
check_reg!(f0_f64, f64, "$f0", "mov.d");

// CHECK-LABEL: reg_ptr:
// CHECK: #APP
// CHECK: move ${{[0-9]+}}, ${{[0-9]+}}
// CHECK: #NO_APP
check!(reg_ptr, ptr, reg, "move");

// CHECK-LABEL: reg_i32:
// CHECK: #APP
// CHECK: move ${{[0-9]+}}, ${{[0-9]+}}
// CHECK: #NO_APP
check!(reg_i32, i32, reg, "move");

// CHECK-LABEL: reg_f32_soft:
// CHECK: #APP
// CHECK: move ${{[0-9]+}}, ${{[0-9]+}}
// CHECK: #NO_APP
check!(reg_f32_soft, f32, reg, "move");

// mips64-LABEL: reg_f64_soft:
// mips64: #APP
// mips64: move ${{[0-9]+}}, ${{[0-9]+}}
// mips64: #NO_APP
#[cfg(mips64)]
check!(reg_f64_soft, f64, reg, "move");

// CHECK-LABEL: reg_i8:
// CHECK: #APP
// CHECK: move ${{[0-9]+}}, ${{[0-9]+}}
// CHECK: #NO_APP
check!(reg_i8, i8, reg, "move");

// CHECK-LABEL: reg_u8:
// CHECK: #APP
// CHECK: move ${{[0-9]+}}, ${{[0-9]+}}
// CHECK: #NO_APP
check!(reg_u8, u8, reg, "move");

// CHECK-LABEL: reg_i16:
// CHECK: #APP
// CHECK: move ${{[0-9]+}}, ${{[0-9]+}}
// CHECK: #NO_APP
check!(reg_i16, i16, reg, "move");

// mips64-LABEL: reg_i64:
// mips64: #APP
// mips64: move ${{[0-9]+}}, ${{[0-9]+}}
// mips64: #NO_APP
#[cfg(mips64)]
check!(reg_i64, i64, reg, "move");

// CHECK-LABEL: r8_ptr:
// CHECK: #APP
// CHECK: move $8, $8
// CHECK: #NO_APP
check_reg!(r8_ptr, ptr, "$8", "move");

// CHECK-LABEL: r8_i32:
// CHECK: #APP
// CHECK: move $8, $8
// CHECK: #NO_APP
check_reg!(r8_i32, i32, "$8", "move");

// CHECK-LABEL: r8_f32:
// CHECK: #APP
// CHECK: move $8, $8
// CHECK: #NO_APP
check_reg!(r8_f32, f32, "$8", "move");

// CHECK-LABEL: r8_i8:
// CHECK: #APP
// CHECK: move $8, $8
// CHECK: #NO_APP
check_reg!(r8_i8, i8, "$8", "move");

// CHECK-LABEL: r8_u8:
// CHECK: #APP
// CHECK: move $8, $8
// CHECK: #NO_APP
check_reg!(r8_u8, u8, "$8", "move");

// CHECK-LABEL: r8_i16:
// CHECK: #APP
// CHECK: move $8, $8
// CHECK: #NO_APP
check_reg!(r8_i16, i16, "$8", "move");
