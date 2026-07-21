//@ add-minicore
//@ revisions: aarch64 arm64ec
//@ assembly-output: emit-asm
//@ [aarch64] compile-flags: --target aarch64-unknown-linux-gnu -C target-feature=+sve
//@ [aarch64] needs-llvm-components: aarch64
//@ [arm64ec] compile-flags: --target arm64ec-pc-windows-msvc
//@ [arm64ec] needs-llvm-components: aarch64
//@ compile-flags: -Zmerge-functions=disabled

#![feature(no_core, repr_simd, f16, f128, rustc_attrs)]
#![crate_type = "rlib"]
#![no_core]
#![allow(asm_sub_register, non_camel_case_types)]

extern crate minicore;
use minicore::*;

type ptr = *mut u8;

#[repr(simd)]
pub struct i8x8([i8; 8]);
#[repr(simd)]
pub struct i16x4([i16; 4]);
#[repr(simd)]
pub struct i32x2([i32; 2]);
#[repr(simd)]
pub struct i64x1([i64; 1]);
#[repr(simd)]
pub struct f16x4([f16; 4]);
#[repr(simd)]
pub struct f32x2([f32; 2]);
#[repr(simd)]
pub struct f64x1([f64; 1]);
#[repr(simd)]
pub struct i8x16([i8; 16]);
#[repr(simd)]
pub struct i16x8([i16; 8]);
#[repr(simd)]
pub struct i32x4([i32; 4]);
#[repr(simd)]
pub struct i64x2([i64; 2]);
#[repr(simd)]
pub struct f16x8([f16; 8]);
#[repr(simd)]
pub struct f32x4([f32; 4]);
#[repr(simd)]
pub struct f64x2([f64; 2]);

#[cfg(target_feature = "sve")]
#[rustc_scalable_vector(16)]
pub struct svint8_t(i8);

#[cfg(target_feature = "sve")]
#[rustc_scalable_vector(8)]
pub struct svint16_t(i16);

#[cfg(target_feature = "sve")]
#[rustc_scalable_vector(4)]
pub struct svint32_t(i32);

#[cfg(target_feature = "sve")]
#[rustc_scalable_vector(2)]
pub struct svint64_t(i64);

#[cfg(target_feature = "sve")]
#[rustc_scalable_vector(8)]
pub struct svfloat16_t(f16);

#[cfg(target_feature = "sve")]
#[rustc_scalable_vector(4)]
pub struct svfloat32_t(f32);

#[cfg(target_feature = "sve")]
#[rustc_scalable_vector(2)]
pub struct svfloat64_t(f64);

#[cfg(target_feature = "sve")]
#[rustc_scalable_vector(16)]
pub struct svbool_t(bool);

impl Copy for i8x8 {}
impl Copy for i16x4 {}
impl Copy for i32x2 {}
impl Copy for i64x1 {}
impl Copy for f16x4 {}
impl Copy for f32x2 {}
impl Copy for f64x1 {}
impl Copy for i8x16 {}
impl Copy for i16x8 {}
impl Copy for i32x4 {}
impl Copy for i64x2 {}
impl Copy for f16x8 {}
impl Copy for f32x4 {}
impl Copy for f64x2 {}

#[cfg(target_feature = "sve")]
impl Copy for svint8_t {}
#[cfg(target_feature = "sve")]
impl Copy for svint16_t {}
#[cfg(target_feature = "sve")]
impl Copy for svint32_t {}
#[cfg(target_feature = "sve")]
impl Copy for svint64_t {}
#[cfg(target_feature = "sve")]
impl Copy for svfloat16_t {}
#[cfg(target_feature = "sve")]
impl Copy for svfloat32_t {}
#[cfg(target_feature = "sve")]
impl Copy for svfloat64_t {}
#[cfg(target_feature = "sve")]
impl Copy for svbool_t {}

extern "C" {
    fn extern_func();
    static extern_static: u8;
}

// CHECK-LABEL: {{("#)?}}sym_fn{{"?}}
// CHECK: //APP
// CHECK: bl extern_func
// CHECK: //NO_APP
#[no_mangle]
pub unsafe fn sym_fn() {
    asm!("bl {}", sym extern_func);
}

// CHECK-LABEL: {{("#)?}}sym_static{{"?}}
// CHECK: //APP
// CHECK: adr x0, extern_static
// CHECK: //NO_APP
#[no_mangle]
pub unsafe fn sym_static() {
    asm!("adr x0, {}", sym extern_static);
}

// Regression test for #75761
// CHECK-LABEL: {{("#)?}}issue_75761{{"?}}
// x29 holds the frame pointer, right next to x30, so ldp/stp happens sometimes
// CHECK: st[[MAY_PAIR:(r|p).*]]x30
// CHECK: //APP
// CHECK: //NO_APP
// CHECK: ld[[MAY_PAIR]]x30
#[no_mangle]
pub unsafe fn issue_75761() {
    asm!("", out("v0") _, out("x30") _);
}

macro_rules! check {
    ($func:ident $ty:ident $class:ident $mov:literal $modifier:literal) => {
        // FIXME(f128): Change back to `$func(x: $ty) -> $ty` once arm64ec can pass and return
        // `f128` without LLVM erroring.
        // LLVM issue: <https://github.com/llvm/llvm-project/issues/94434>
        #[no_mangle]
        pub unsafe fn $func(inp: &$ty, out: &mut $ty) {
            let x = *inp;
            let y;
            asm!(
                concat!($mov, " {:", $modifier, "}, {:", $modifier, "}"),
                out($class) y,
                in($class) x
            );
            *out = y;
        }
    };
}

macro_rules! check_sve {
    ($func:ident $ty:ident $class:ident $suffix:literal $zm:literal) => {
        #[no_mangle]
        pub unsafe fn $func(inp: &$ty, pred: &svbool_t) -> $ty {
            let x = *inp;
            let z = *pred;
            let y;
            asm!(
                concat!("mov {0}.", $suffix, ", {1}/", $zm, ", {2}.", $suffix),
                out($class) y,
                in(preg) z,
                in($class) x
            );
            y
        }
    };
}

macro_rules! check_reg {
    ($func:ident $ty:ident $reg:tt $mov:literal) => {
        // FIXME(f128): See FIXME in `check!`
        #[no_mangle]
        pub unsafe fn $func(inp: &$ty, out: &mut $ty) {
            let x = *inp;
            let y;
            asm!(concat!($mov, " ", $reg, ", ", $reg), lateout($reg) y, in($reg) x);
            *out = y;
        }
    };
}

macro_rules! check_reg_sve {
    ($func:ident $ty:ident $reg:tt $suffix:literal $zm:literal) => {
        #[no_mangle]
        pub unsafe fn $func(inp: &$ty, pred: &svbool_t) -> $ty {
            let x = *inp;
            let z = *pred;
            let y;
            asm!(
                concat!("mov ", $reg, ".", $suffix, ", {}/", $zm, ", ", $reg, ".", $suffix),
                in(preg) z,
                lateout($reg) y,
                in($reg) x
            );
            y
        }
    };
}

// CHECK-LABEL: {{("#)?}}reg_i8{{"?}}
// CHECK: //APP
// CHECK: mov x{{[0-9]+}}, x{{[0-9]+}}
// CHECK: //NO_APP
check!(reg_i8 i8 reg "mov" "");

// CHECK-LABEL: {{("#)?}}reg_i16{{"?}}
// CHECK: //APP
// CHECK: mov x{{[0-9]+}}, x{{[0-9]+}}
// CHECK: //NO_APP
check!(reg_i16 i16 reg "mov" "");

// CHECK-LABEL: {{("#)?}}reg_f16{{"?}}
// CHECK: //APP
// CHECK: mov x{{[0-9]+}}, x{{[0-9]+}}
// CHECK: //NO_APP
check!(reg_f16 f16 reg "mov" "");

// CHECK-LABEL: {{("#)?}}reg_i32{{"?}}
// CHECK: //APP
// CHECK: mov x{{[0-9]+}}, x{{[0-9]+}}
// CHECK: //NO_APP
check!(reg_i32 i32 reg "mov" "");

// CHECK-LABEL: {{("#)?}}reg_f32{{"?}}
// CHECK: //APP
// CHECK: mov x{{[0-9]+}}, x{{[0-9]+}}
// CHECK: //NO_APP
check!(reg_f32 f32 reg "mov" "");

// CHECK-LABEL: {{("#)?}}reg_i64{{"?}}
// CHECK: //APP
// CHECK: mov x{{[0-9]+}}, x{{[0-9]+}}
// CHECK: //NO_APP
check!(reg_i64 i64 reg "mov" "");

// CHECK-LABEL: {{("#)?}}reg_f64{{"?}}
// CHECK: //APP
// CHECK: mov x{{[0-9]+}}, x{{[0-9]+}}
// CHECK: //NO_APP
check!(reg_f64 f64 reg "mov" "");

// CHECK-LABEL: {{("#)?}}reg_ptr{{"?}}
// CHECK: //APP
// CHECK: mov x{{[0-9]+}}, x{{[0-9]+}}
// CHECK: //NO_APP
check!(reg_ptr ptr reg "mov" "");

// CHECK-LABEL: {{("#)?}}vreg_i8{{"?}}
// CHECK: //APP
// CHECK: fmov s{{[0-9]+}}, s{{[0-9]+}}
// CHECK: //NO_APP
check!(vreg_i8 i8 vreg "fmov" "s");

// CHECK-LABEL: {{("#)?}}vreg_i16{{"?}}
// CHECK: //APP
// CHECK: fmov s{{[0-9]+}}, s{{[0-9]+}}
// CHECK: //NO_APP
check!(vreg_i16 i16 vreg "fmov" "s");

// CHECK-LABEL: {{("#)?}}vreg_f16{{"?}}
// CHECK: //APP
// CHECK: fmov s{{[0-9]+}}, s{{[0-9]+}}
// CHECK: //NO_APP
check!(vreg_f16 f16 vreg "fmov" "s");

// CHECK-LABEL: {{("#)?}}vreg_i32{{"?}}
// CHECK: //APP
// CHECK: fmov s{{[0-9]+}}, s{{[0-9]+}}
// CHECK: //NO_APP
check!(vreg_i32 i32 vreg "fmov" "s");

// CHECK-LABEL: {{("#)?}}vreg_f32{{"?}}
// CHECK: //APP
// CHECK: fmov s{{[0-9]+}}, s{{[0-9]+}}
// CHECK: //NO_APP
check!(vreg_f32 f32 vreg "fmov" "s");

// CHECK-LABEL: {{("#)?}}vreg_i64{{"?}}
// CHECK: //APP
// CHECK: fmov s{{[0-9]+}}, s{{[0-9]+}}
// CHECK: //NO_APP
check!(vreg_i64 i64 vreg "fmov" "s");

// CHECK-LABEL: {{("#)?}}vreg_f64{{"?}}
// CHECK: //APP
// CHECK: fmov s{{[0-9]+}}, s{{[0-9]+}}
// CHECK: //NO_APP
check!(vreg_f64 f64 vreg "fmov" "s");

// CHECK-LABEL: {{("#)?}}vreg_f128{{"?}}
// CHECK: //APP
// CHECK: fmov s{{[0-9]+}}, s{{[0-9]+}}
// CHECK: //NO_APP
check!(vreg_f128 f128 vreg "fmov" "s");

// CHECK-LABEL: {{("#)?}}vreg_ptr{{"?}}
// CHECK: //APP
// CHECK: fmov s{{[0-9]+}}, s{{[0-9]+}}
// CHECK: //NO_APP
check!(vreg_ptr ptr vreg "fmov" "s");

// CHECK-LABEL: {{("#)?}}vreg_i8x8{{"?}}
// CHECK: //APP
// CHECK: fmov s{{[0-9]+}}, s{{[0-9]+}}
// CHECK: //NO_APP
check!(vreg_i8x8 i8x8 vreg "fmov" "s");

// CHECK-LABEL: {{("#)?}}vreg_i16x4{{"?}}
// CHECK: //APP
// CHECK: fmov s{{[0-9]+}}, s{{[0-9]+}}
// CHECK: //NO_APP
check!(vreg_i16x4 i16x4 vreg "fmov" "s");

// CHECK-LABEL: {{("#)?}}vreg_i32x2{{"?}}
// CHECK: //APP
// CHECK: fmov s{{[0-9]+}}, s{{[0-9]+}}
// CHECK: //NO_APP
check!(vreg_i32x2 i32x2 vreg "fmov" "s");

// CHECK-LABEL: {{("#)?}}vreg_i64x1{{"?}}
// CHECK: //APP
// CHECK: fmov s{{[0-9]+}}, s{{[0-9]+}}
// CHECK: //NO_APP
check!(vreg_i64x1 i64x1 vreg "fmov" "s");

// CHECK-LABEL: {{("#)?}}vreg_f16x4{{"?}}
// CHECK: //APP
// CHECK: fmov s{{[0-9]+}}, s{{[0-9]+}}
// CHECK: //NO_APP
check!(vreg_f16x4 f16x4 vreg "fmov" "s");

// CHECK-LABEL: {{("#)?}}vreg_f32x2{{"?}}
// CHECK: //APP
// CHECK: fmov s{{[0-9]+}}, s{{[0-9]+}}
// CHECK: //NO_APP
check!(vreg_f32x2 f32x2 vreg "fmov" "s");

// CHECK-LABEL: {{("#)?}}vreg_f64x1{{"?}}
// CHECK: //APP
// CHECK: fmov s{{[0-9]+}}, s{{[0-9]+}}
// CHECK: //NO_APP
check!(vreg_f64x1 f64x1 vreg "fmov" "s");

// CHECK-LABEL: {{("#)?}}vreg_i8x16{{"?}}
// CHECK: //APP
// CHECK: fmov s{{[0-9]+}}, s{{[0-9]+}}
// CHECK: //NO_APP
check!(vreg_i8x16 i8x16 vreg "fmov" "s");

// CHECK-LABEL: {{("#)?}}vreg_i16x8{{"?}}
// CHECK: //APP
// CHECK: fmov s{{[0-9]+}}, s{{[0-9]+}}
// CHECK: //NO_APP
check!(vreg_i16x8 i16x8 vreg "fmov" "s");

// CHECK-LABEL: {{("#)?}}vreg_i32x4{{"?}}
// CHECK: //APP
// CHECK: fmov s{{[0-9]+}}, s{{[0-9]+}}
// CHECK: //NO_APP
check!(vreg_i32x4 i32x4 vreg "fmov" "s");

// CHECK-LABEL: {{("#)?}}vreg_i64x2{{"?}}
// CHECK: //APP
// CHECK: fmov s{{[0-9]+}}, s{{[0-9]+}}
// CHECK: //NO_APP
check!(vreg_i64x2 i64x2 vreg "fmov" "s");

// CHECK-LABEL: {{("#)?}}vreg_f16x8{{"?}}
// CHECK: //APP
// CHECK: fmov s{{[0-9]+}}, s{{[0-9]+}}
// CHECK: //NO_APP
check!(vreg_f16x8 f16x8 vreg "fmov" "s");

// CHECK-LABEL: {{("#)?}}vreg_f32x4{{"?}}
// CHECK: //APP
// CHECK: fmov s{{[0-9]+}}, s{{[0-9]+}}
// CHECK: //NO_APP
check!(vreg_f32x4 f32x4 vreg "fmov" "s");

// CHECK-LABEL: {{("#)?}}vreg_f64x2{{"?}}
// CHECK: //APP
// CHECK: fmov s{{[0-9]+}}, s{{[0-9]+}}
// CHECK: //NO_APP
check!(vreg_f64x2 f64x2 vreg "fmov" "s");

// CHECK-LABEL: {{("#)?}}vreg_low16_i8{{"?}}
// CHECK: //APP
// CHECK: fmov s{{[0-9]+}}, s{{[0-9]+}}
// CHECK: //NO_APP
check!(vreg_low16_i8 i8 vreg_low16 "fmov" "s");

// CHECK-LABEL: {{("#)?}}vreg_low16_i16{{"?}}
// CHECK: //APP
// CHECK: fmov s{{[0-9]+}}, s{{[0-9]+}}
// CHECK: //NO_APP
check!(vreg_low16_i16 i16 vreg_low16 "fmov" "s");

// CHECK-LABEL: {{("#)?}}vreg_low16_f16{{"?}}
// CHECK: //APP
// CHECK: fmov s{{[0-9]+}}, s{{[0-9]+}}
// CHECK: //NO_APP
check!(vreg_low16_f16 f16 vreg_low16 "fmov" "s");

// CHECK-LABEL: {{("#)?}}vreg_low16_f32{{"?}}
// CHECK: //APP
// CHECK: fmov s{{[0-9]+}}, s{{[0-9]+}}
// CHECK: //NO_APP
check!(vreg_low16_f32 f32 vreg_low16 "fmov" "s");

// CHECK-LABEL: {{("#)?}}vreg_low16_i64{{"?}}
// CHECK: //APP
// CHECK: fmov s{{[0-9]+}}, s{{[0-9]+}}
// CHECK: //NO_APP
check!(vreg_low16_i64 i64 vreg_low16 "fmov" "s");

// CHECK-LABEL: {{("#)?}}vreg_low16_f64{{"?}}
// CHECK: //APP
// CHECK: fmov s{{[0-9]+}}, s{{[0-9]+}}
// CHECK: //NO_APP
check!(vreg_low16_f64 f64 vreg_low16 "fmov" "s");

// CHECK-LABEL: {{("#)?}}vreg_low16_f128{{"?}}
// CHECK: //APP
// CHECK: fmov s{{[0-9]+}}, s{{[0-9]+}}
// CHECK: //NO_APP
check!(vreg_low16_f128 f128 vreg_low16 "fmov" "s");

// CHECK-LABEL: {{("#)?}}vreg_low16_ptr{{"?}}
// CHECK: //APP
// CHECK: fmov s{{[0-9]+}}, s{{[0-9]+}}
// CHECK: //NO_APP
check!(vreg_low16_ptr ptr vreg_low16 "fmov" "s");

// CHECK-LABEL: {{("#)?}}vreg_low16_i8x8{{"?}}
// CHECK: //APP
// CHECK: fmov s{{[0-9]+}}, s{{[0-9]+}}
// CHECK: //NO_APP
check!(vreg_low16_i8x8 i8x8 vreg_low16 "fmov" "s");

// CHECK-LABEL: {{("#)?}}vreg_low16_i16x4{{"?}}
// CHECK: //APP
// CHECK: fmov s{{[0-9]+}}, s{{[0-9]+}}
// CHECK: //NO_APP
check!(vreg_low16_i16x4 i16x4 vreg_low16 "fmov" "s");

// CHECK-LABEL: {{("#)?}}vreg_low16_i32x2{{"?}}
// CHECK: //APP
// CHECK: fmov s{{[0-9]+}}, s{{[0-9]+}}
// CHECK: //NO_APP
check!(vreg_low16_i32x2 i32x2 vreg_low16 "fmov" "s");

// CHECK-LABEL: {{("#)?}}vreg_low16_i64x1{{"?}}
// CHECK: //APP
// CHECK: fmov s{{[0-9]+}}, s{{[0-9]+}}
// CHECK: //NO_APP
check!(vreg_low16_i64x1 i64x1 vreg_low16 "fmov" "s");

// CHECK-LABEL: {{("#)?}}vreg_low16_f16x4{{"?}}
// CHECK: //APP
// CHECK: fmov s{{[0-9]+}}, s{{[0-9]+}}
// CHECK: //NO_APP
check!(vreg_low16_f16x4 f16x4 vreg_low16 "fmov" "s");

// CHECK-LABEL: {{("#)?}}vreg_low16_f32x2{{"?}}
// CHECK: //APP
// CHECK: fmov s{{[0-9]+}}, s{{[0-9]+}}
// CHECK: //NO_APP
check!(vreg_low16_f32x2 f32x2 vreg_low16 "fmov" "s");

// CHECK-LABEL: {{("#)?}}vreg_low16_f64x1{{"?}}
// CHECK: //APP
// CHECK: fmov s{{[0-9]+}}, s{{[0-9]+}}
// CHECK: //NO_APP
check!(vreg_low16_f64x1 f64x1 vreg_low16 "fmov" "s");

// CHECK-LABEL: {{("#)?}}vreg_low16_i8x16{{"?}}
// CHECK: //APP
// CHECK: fmov s{{[0-9]+}}, s{{[0-9]+}}
// CHECK: //NO_APP
check!(vreg_low16_i8x16 i8x16 vreg_low16 "fmov" "s");

// CHECK-LABEL: {{("#)?}}vreg_low16_i16x8{{"?}}
// CHECK: //APP
// CHECK: fmov s{{[0-9]+}}, s{{[0-9]+}}
// CHECK: //NO_APP
check!(vreg_low16_i16x8 i16x8 vreg_low16 "fmov" "s");

// CHECK-LABEL: {{("#)?}}vreg_low16_i32x4{{"?}}
// CHECK: //APP
// CHECK: fmov s{{[0-9]+}}, s{{[0-9]+}}
// CHECK: //NO_APP
check!(vreg_low16_i32x4 i32x4 vreg_low16 "fmov" "s");

// CHECK-LABEL: {{("#)?}}vreg_low16_i64x2{{"?}}
// CHECK: //APP
// CHECK: fmov s{{[0-9]+}}, s{{[0-9]+}}
// CHECK: //NO_APP
check!(vreg_low16_i64x2 i64x2 vreg_low16 "fmov" "s");

// CHECK-LABEL: {{("#)?}}vreg_low16_f16x8{{"?}}
// CHECK: //APP
// CHECK: fmov s{{[0-9]+}}, s{{[0-9]+}}
// CHECK: //NO_APP
check!(vreg_low16_f16x8 f16x8 vreg_low16 "fmov" "s");

// CHECK-LABEL: {{("#)?}}vreg_low16_f32x4{{"?}}
// CHECK: //APP
// CHECK: fmov s{{[0-9]+}}, s{{[0-9]+}}
// CHECK: //NO_APP
check!(vreg_low16_f32x4 f32x4 vreg_low16 "fmov" "s");

// CHECK-LABEL: {{("#)?}}vreg_low16_f64x2{{"?}}
// CHECK: //APP
// CHECK: fmov s{{[0-9]+}}, s{{[0-9]+}}
// CHECK: //NO_APP
check!(vreg_low16_f64x2 f64x2 vreg_low16 "fmov" "s");

#[cfg(target_feature = "sve")]
// aarch64-LABEL: {{("#)?}}zreg_i8{{"?}}
// aarch64: //APP
// aarch64: mov z{{[0-9]+}}.b, p0/m, z{{[0-9]+}}.b
// aarch64: //NO_APP
check_sve!(zreg_i8 svint8_t zreg "b" "m");

#[cfg(target_feature = "sve")]
// aarch64-LABEL: {{("#)?}}zreg_i16{{"?}}
// aarch64: //APP
// aarch64: mov z{{[0-9]+}}.h, p0/m, z{{[0-9]+}}.h
// aarch64: //NO_APP
check_sve!(zreg_i16 svint16_t zreg "h" "m");

#[cfg(target_feature = "sve")]
// aarch64-LABEL: {{("#)?}}zreg_f16{{"?}}
// aarch64: //APP
// aarch64: mov z{{[0-9]+}}.h, p0/m, z{{[0-9]+}}.h
// aarch64: //NO_APP
check_sve!(zreg_f16 svfloat16_t zreg "h" "m");

#[cfg(target_feature = "sve")]
// aarch64-LABEL: {{("#)?}}zreg_i32{{"?}}
// aarch64: //APP
// aarch64: mov z{{[0-9]+}}.s, p0/m, z{{[0-9]+}}.s
// aarch64: //NO_APP
check_sve!(zreg_i32 svint32_t zreg "s" "m");

#[cfg(target_feature = "sve")]
// aarch64-LABEL: {{("#)?}}zreg_f32{{"?}}
// aarch64: //APP
// aarch64: mov z{{[0-9]+}}.s, p0/m, z{{[0-9]+}}.s
// aarch64: //NO_APP
check_sve!(zreg_f32 svfloat32_t zreg "s" "m");

#[cfg(target_feature = "sve")]
// aarch64-LABEL: {{("#)?}}zreg_i64{{"?}}
// aarch64: //APP
// aarch64: mov z{{[0-9]+}}.d, p0/m, z{{[0-9]+}}.d
// aarch64: //NO_APP
check_sve!(zreg_i64 svint64_t zreg "d" "m");

#[cfg(target_feature = "sve")]
// aarch64-LABEL: {{("#)?}}zreg_f64{{"?}}
// aarch64: //APP
// aarch64: mov z{{[0-9]+}}.d, p0/m, z{{[0-9]+}}.d
// aarch64: //NO_APP
check_sve!(zreg_f64 svfloat64_t zreg "d" "m");

#[cfg(target_feature = "sve")]
// aarch64-LABEL: {{("#)?}}zreg_bool{{"?}}
// aarch64: //APP
// aarch64: mov p{{[0-9]+}}.b, p{{[0-9]+}}/z, p{{[0-9]+}}.b
// aarch64: //NO_APP
check_sve!(zreg_bool svbool_t preg "b" "z");

// CHECK-LABEL: {{("#)?}}x0_i8{{"?}}
// CHECK: //APP
// CHECK: mov x{{[0-9]+}}, x{{[0-9]+}}
// CHECK: //NO_APP
check_reg!(x0_i8 i8 "x0" "mov");

// CHECK-LABEL: {{("#)?}}x0_i16{{"?}}
// CHECK: //APP
// CHECK: mov x{{[0-9]+}}, x{{[0-9]+}}
// CHECK: //NO_APP
check_reg!(x0_i16 i16 "x0" "mov");

// CHECK-LABEL: {{("#)?}}x0_f16{{"?}}
// CHECK: //APP
// CHECK: mov x{{[0-9]+}}, x{{[0-9]+}}
// CHECK: //NO_APP
check_reg!(x0_f16 f16 "x0" "mov");

// CHECK-LABEL: {{("#)?}}x0_i32{{"?}}
// CHECK: //APP
// CHECK: mov x{{[0-9]+}}, x{{[0-9]+}}
// CHECK: //NO_APP
check_reg!(x0_i32 i32 "x0" "mov");

// CHECK-LABEL: {{("#)?}}x0_f32{{"?}}
// CHECK: //APP
// CHECK: mov x{{[0-9]+}}, x{{[0-9]+}}
// CHECK: //NO_APP
check_reg!(x0_f32 f32 "x0" "mov");

// CHECK-LABEL: {{("#)?}}x0_i64{{"?}}
// CHECK: //APP
// CHECK: mov x{{[0-9]+}}, x{{[0-9]+}}
// CHECK: //NO_APP
check_reg!(x0_i64 i64 "x0" "mov");

// CHECK-LABEL: {{("#)?}}x0_f64{{"?}}
// CHECK: //APP
// CHECK: mov x{{[0-9]+}}, x{{[0-9]+}}
// CHECK: //NO_APP
check_reg!(x0_f64 f64 "x0" "mov");

// CHECK-LABEL: {{("#)?}}x0_ptr{{"?}}
// CHECK: //APP
// CHECK: mov x{{[0-9]+}}, x{{[0-9]+}}
// CHECK: //NO_APP
check_reg!(x0_ptr ptr "x0" "mov");

// CHECK-LABEL: {{("#)?}}v0_i8{{"?}}
// CHECK: //APP
// CHECK: fmov s0, s0
// CHECK: //NO_APP
check_reg!(v0_i8 i8 "s0" "fmov");

// CHECK-LABEL: {{("#)?}}v0_i16{{"?}}
// CHECK: //APP
// CHECK: fmov s0, s0
// CHECK: //NO_APP
check_reg!(v0_i16 i16 "s0" "fmov");

// CHECK-LABEL: {{("#)?}}v0_f16{{"?}}
// CHECK: //APP
// CHECK: fmov s0, s0
// CHECK: //NO_APP
check_reg!(v0_f16 f16 "s0" "fmov");

// CHECK-LABEL: {{("#)?}}v0_i32{{"?}}
// CHECK: //APP
// CHECK: fmov s0, s0
// CHECK: //NO_APP
check_reg!(v0_i32 i32 "s0" "fmov");

// CHECK-LABEL: {{("#)?}}v0_f32{{"?}}
// CHECK: //APP
// CHECK: fmov s0, s0
// CHECK: //NO_APP
check_reg!(v0_f32 f32 "s0" "fmov");

// CHECK-LABEL: {{("#)?}}v0_i64{{"?}}
// CHECK: //APP
// CHECK: fmov s0, s0
// CHECK: //NO_APP
check_reg!(v0_i64 i64 "s0" "fmov");

// CHECK-LABEL: {{("#)?}}v0_f64{{"?}}
// CHECK: //APP
// CHECK: fmov s0, s0
// CHECK: //NO_APP
check_reg!(v0_f64 f64 "s0" "fmov");

// CHECK-LABEL: {{("#)?}}v0_f128{{"?}}
// CHECK: //APP
// CHECK: fmov s0, s0
// CHECK: //NO_APP
check_reg!(v0_f128 f128 "s0" "fmov");

#[cfg(target_feature = "sve")]
// aarch64-LABEL: {{("#)?}}z0_i8{{"?}}
// aarch64: //APP
// aarch64: mov z0.b, p0/m, z0.b
// aarch64: //NO_APP
check_reg_sve!(z0_i8 svint8_t "z0" "b" "m");

#[cfg(target_feature = "sve")]
// aarch64-LABEL: {{("#)?}}z0_i16{{"?}}
// aarch64: //APP
// aarch64: mov z0.h, p0/m, z0.h
// aarch64: //NO_APP
check_reg_sve!(z0_i16 svint16_t "z0" "h" "m");

#[cfg(target_feature = "sve")]
// aarch64-LABEL: {{("#)?}}z0_f16{{"?}}
// aarch64: //APP
// aarch64: mov z0.h, p0/m, z0.h
// aarch64: //NO_APP
check_reg_sve!(z0_f16 svfloat16_t "z0" "h" "m");

#[cfg(target_feature = "sve")]
// aarch64-LABEL: {{("#)?}}z0_i32{{"?}}
// aarch64: //APP
// aarch64: mov z0.s, p0/m, z0.s
// aarch64: //NO_APP
check_reg_sve!(z0_i32 svint32_t "z0" "s" "m");

#[cfg(target_feature = "sve")]
// aarch64-LABEL: {{("#)?}}z0_f32{{"?}}
// aarch64: //APP
// aarch64: mov z0.s, p0/m, z0.s
// aarch64: //NO_APP
check_reg_sve!(z0_f32 svfloat32_t "z0" "s" "m");

#[cfg(target_feature = "sve")]
// aarch64-LABEL: {{("#)?}}z0_i64{{"?}}
// aarch64: //APP
// aarch64: mov z0.d, p0/m, z0.d
// aarch64: //NO_APP
check_reg_sve!(z0_i64 svint64_t "z0" "d" "m");

#[cfg(target_feature = "sve")]
// aarch64-LABEL: {{("#)?}}z0_f64{{"?}}
// aarch64: //APP
// aarch64: mov z0.d, p0/m, z0.d
// aarch64: //NO_APP
check_reg_sve!(z0_f64 svfloat64_t "z0" "d" "m");

#[cfg(target_feature = "sve")]
// aarch64-LABEL: {{("#)?}}p0_bool{{"?}}
// aarch64: //APP
// aarch64: mov p0.b, p1/z, p0.b
// aarch64: //NO_APP
check_reg_sve!(p0_bool svbool_t "p0" "b" "z");

// CHECK-LABEL: {{("#)?}}v0_ptr{{"?}}
// CHECK: //APP
// CHECK: fmov s0, s0
// CHECK: //NO_APP
check_reg!(v0_ptr ptr "s0" "fmov");

// CHECK-LABEL: {{("#)?}}v0_i8x8{{"?}}
// CHECK: //APP
// CHECK: fmov s0, s0
// CHECK: //NO_APP
check_reg!(v0_i8x8 i8x8 "s0" "fmov");

// CHECK-LABEL: {{("#)?}}v0_i16x4{{"?}}
// CHECK: //APP
// CHECK: fmov s0, s0
// CHECK: //NO_APP
check_reg!(v0_i16x4 i16x4 "s0" "fmov");

// CHECK-LABEL: {{("#)?}}v0_i32x2{{"?}}
// CHECK: //APP
// CHECK: fmov s0, s0
// CHECK: //NO_APP
check_reg!(v0_i32x2 i32x2 "s0" "fmov");

// CHECK-LABEL: {{("#)?}}v0_i64x1{{"?}}
// CHECK: //APP
// CHECK: fmov s0, s0
// CHECK: //NO_APP
check_reg!(v0_i64x1 i64x1 "s0" "fmov");

// CHECK-LABEL: {{("#)?}}v0_f16x4{{"?}}
// CHECK: //APP
// CHECK: fmov s0, s0
// CHECK: //NO_APP
check_reg!(v0_f16x4 f16x4 "s0" "fmov");

// CHECK-LABEL: {{("#)?}}v0_f32x2{{"?}}
// CHECK: //APP
// CHECK: fmov s0, s0
// CHECK: //NO_APP
check_reg!(v0_f32x2 f32x2 "s0" "fmov");

// CHECK-LABEL: {{("#)?}}v0_f64x1{{"?}}
// CHECK: //APP
// CHECK: fmov s0, s0
// CHECK: //NO_APP
check_reg!(v0_f64x1 f64x1 "s0" "fmov");

// CHECK-LABEL: {{("#)?}}v0_i8x16{{"?}}
// CHECK: //APP
// CHECK: fmov s0, s0
// CHECK: //NO_APP
check_reg!(v0_i8x16 i8x16 "s0" "fmov");

// CHECK-LABEL: {{("#)?}}v0_i16x8{{"?}}
// CHECK: //APP
// CHECK: fmov s0, s0
// CHECK: //NO_APP
check_reg!(v0_i16x8 i16x8 "s0" "fmov");

// CHECK-LABEL: {{("#)?}}v0_i32x4{{"?}}
// CHECK: //APP
// CHECK: fmov s0, s0
// CHECK: //NO_APP
check_reg!(v0_i32x4 i32x4 "s0" "fmov");

// CHECK-LABEL: {{("#)?}}v0_i64x2{{"?}}
// CHECK: //APP
// CHECK: fmov s0, s0
// CHECK: //NO_APP
check_reg!(v0_i64x2 i64x2 "s0" "fmov");

// CHECK-LABEL: {{("#)?}}v0_f16x8{{"?}}
// CHECK: //APP
// CHECK: fmov s0, s0
// CHECK: //NO_APP
check_reg!(v0_f16x8 f16x8 "s0" "fmov");

// CHECK-LABEL: {{("#)?}}v0_f32x4{{"?}}
// CHECK: //APP
// CHECK: fmov s0, s0
// CHECK: //NO_APP
check_reg!(v0_f32x4 f32x4 "s0" "fmov");

// CHECK-LABEL: {{("#)?}}v0_f64x2{{"?}}
// CHECK: //APP
// CHECK: fmov s0, s0
// CHECK: //NO_APP
check_reg!(v0_f64x2 f64x2 "s0" "fmov");
