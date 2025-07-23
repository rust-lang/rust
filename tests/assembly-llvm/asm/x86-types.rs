//@ add-core-stubs
//@ revisions: x86_64 i686
//@ assembly-output: emit-asm
//@[x86_64] compile-flags: --target x86_64-unknown-linux-gnu
//@[x86_64] needs-llvm-components: x86
//@[i686] compile-flags: --target i686-unknown-linux-gnu
//@[i686] needs-llvm-components: x86
//@ compile-flags: -C llvm-args=--x86-asm-syntax=intel
//@ compile-flags: -C target-feature=+avx512bw
//@ compile-flags: -Zmerge-functions=disabled

#![feature(no_core, repr_simd, f16, f128)]
#![crate_type = "rlib"]
#![no_core]
#![allow(asm_sub_register, non_camel_case_types)]

extern crate minicore;
use minicore::*;

type ptr = *mut u8;

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

#[repr(simd)]
pub struct i8x32([i8; 32]);
#[repr(simd)]
pub struct i16x16([i16; 16]);
#[repr(simd)]
pub struct i32x8([i32; 8]);
#[repr(simd)]
pub struct i64x4([i64; 4]);
#[repr(simd)]
pub struct f16x16([f16; 16]);
#[repr(simd)]
pub struct f32x8([f32; 8]);
#[repr(simd)]
pub struct f64x4([f64; 4]);

#[repr(simd)]
pub struct i8x64([i8; 64]);
#[repr(simd)]
pub struct i16x32([i16; 32]);
#[repr(simd)]
pub struct i32x16([i32; 16]);
#[repr(simd)]
pub struct i64x8([i64; 8]);
#[repr(simd)]
pub struct f16x32([f16; 32]);
#[repr(simd)]
pub struct f32x16([f32; 16]);
#[repr(simd)]
pub struct f64x8([f64; 8]);

macro_rules! impl_copy {
    ($($ty:ident)*) => {
        $(
            impl Copy for $ty {}
        )*
    };
}

impl_copy!(
    i8x16 i16x8 i32x4 i64x2 f16x8 f32x4 f64x2
    i8x32 i16x16 i32x8 i64x4 f16x16 f32x8 f64x4
    i8x64 i16x32 i32x16 i64x8 f16x32 f32x16 f64x8
);

extern "C" {
    fn extern_func();
    static extern_static: u8;
}

// CHECK-LABEL: sym_fn:
// CHECK: #APP
// CHECK: call extern_func
// CHECK: #NO_APP
#[no_mangle]
pub unsafe fn sym_fn() {
    asm!("call {}", sym extern_func);
}

// CHECK-LABEL: sym_static:
// CHECK: #APP
// CHECK: mov al, byte ptr [extern_static]
// CHECK: #NO_APP
#[no_mangle]
pub unsafe fn sym_static() {
    asm!("mov al, byte ptr [{}]", sym extern_static);
}

macro_rules! check {
    ($func:ident $ty:ident $class:ident $mov:literal) => {
        #[no_mangle]
        pub unsafe fn $func(x: $ty) -> $ty {
            let y;
            asm!(concat!($mov, " {}, {}"), lateout($class) y, in($class) x);
            y
        }
    };
}

macro_rules! check_reg {
    ($func:ident $ty:ident $reg:tt $mov:literal) => {
        #[no_mangle]
        pub unsafe fn $func(x: $ty) -> $ty {
            let y;
            asm!(concat!($mov, " ", $reg, ", ", $reg), lateout($reg) y, in($reg) x);
            y
        }
    };
}

// CHECK-LABEL: reg_i16:
// CHECK: #APP
// x86_64: mov r{{[a-z0-9]+}}, r{{[a-z0-9]+}}
// i686: mov e{{[a-z0-9]+}}, e{{[a-z0-9]+}}
// CHECK: #NO_APP
check!(reg_i16 i16 reg "mov");

// CHECK-LABEL: reg_f16:
// CHECK: #APP
// x86_64: mov r{{[a-z0-9]+}}, r{{[a-z0-9]+}}
// i686: mov e{{[a-z0-9]+}}, e{{[a-z0-9]+}}
// CHECK: #NO_APP
check!(reg_f16 f16 reg "mov");

// CHECK-LABEL: reg_i32:
// CHECK: #APP
// x86_64: mov r{{[a-z0-9]+}}, r{{[a-z0-9]+}}
// i686: mov e{{[a-z0-9]+}}, e{{[a-z0-9]+}}
// CHECK: #NO_APP
check!(reg_i32 i32 reg "mov");

// CHECK-LABEL: reg_f32:
// CHECK: #APP
// x86_64: mov r{{[a-z0-9]+}}, r{{[a-z0-9]+}}
// i686: mov e{{[a-z0-9]+}}, e{{[a-z0-9]+}}
// CHECK: #NO_APP
check!(reg_f32 f32 reg "mov");

// x86_64-LABEL: reg_i64:
// x86_64: #APP
// x86_64: mov r{{[a-z0-9]+}}, r{{[a-z0-9]+}}
// x86_64: #NO_APP
#[cfg(x86_64)]
check!(reg_i64 i64 reg "mov");

// x86_64-LABEL: reg_f64:
// x86_64: #APP
// x86_64: mov r{{[a-z0-9]+}}, r{{[a-z0-9]+}}
// x86_64: #NO_APP
#[cfg(x86_64)]
check!(reg_f64 f64 reg "mov");

// CHECK-LABEL: reg_ptr:
// CHECK: #APP
// x86_64: mov r{{[a-z0-9]+}}, r{{[a-z0-9]+}}
// i686: mov e{{[a-z0-9]+}}, e{{[a-z0-9]+}}
// CHECK: #NO_APP
check!(reg_ptr ptr reg "mov");

// CHECK-LABEL: reg_abcd_i16:
// CHECK: #APP
// x86_64: mov r{{[a-z0-9]+}}, r{{[a-z0-9]+}}
// i686: mov e{{[a-z0-9]+}}, e{{[a-z0-9]+}}
// CHECK: #NO_APP
check!(reg_abcd_i16 i16 reg_abcd "mov");

// CHECK-LABEL: reg_abcd_f16:
// CHECK: #APP
// x86_64: mov r{{[a-z0-9]+}}, r{{[a-z0-9]+}}
// i686: mov e{{[a-z0-9]+}}, e{{[a-z0-9]+}}
// CHECK: #NO_APP
check!(reg_abcd_f16 f16 reg_abcd "mov");

// CHECK-LABEL: reg_abcd_i32:
// CHECK: #APP
// x86_64: mov r{{[a-z0-9]+}}, r{{[a-z0-9]+}}
// i686: mov e{{[a-z0-9]+}}, e{{[a-z0-9]+}}
// CHECK: #NO_APP
check!(reg_abcd_i32 i32 reg_abcd "mov");

// CHECK-LABEL: reg_abcd_f32:
// CHECK: #APP
// x86_64: mov r{{[a-z0-9]+}}, r{{[a-z0-9]+}}
// i686: mov e{{[a-z0-9]+}}, e{{[a-z0-9]+}}
// CHECK: #NO_APP
check!(reg_abcd_f32 f32 reg_abcd "mov");

// x86_64-LABEL: reg_abcd_i64:
// x86_64: #APP
// x86_64: mov r{{[a-z0-9]+}}, r{{[a-z0-9]+}}
// x86_64: #NO_APP
#[cfg(x86_64)]
check!(reg_abcd_i64 i64 reg_abcd "mov");

// x86_64-LABEL: reg_abcd_f64:
// x86_64: #APP
// x86_64: mov r{{[a-z0-9]+}}, r{{[a-z0-9]+}}
// x86_64: #NO_APP
#[cfg(x86_64)]
check!(reg_abcd_f64 f64 reg_abcd "mov");

// CHECK-LABEL: reg_abcd_ptr:
// CHECK: #APP
// x86_64: mov r{{[a-z0-9]+}}, r{{[a-z0-9]+}}
// i686: mov e{{[a-z0-9]+}}, e{{[a-z0-9]+}}
// CHECK: #NO_APP
check!(reg_abcd_ptr ptr reg_abcd "mov");

// CHECK-LABEL: reg_byte:
// CHECK: #APP
// CHECK: mov {{[a-z0-9]+}}, {{[a-z0-9]+}}
// CHECK: #NO_APP
check!(reg_byte i8 reg_byte "mov");

// CHECK-LABEL: xmm_reg_f16:
// CHECK: #APP
// CHECK: movaps xmm{{[0-9]+}}, xmm{{[0-9]+}}
// CHECK: #NO_APP
check!(xmm_reg_f16 f16 xmm_reg "movaps");

// CHECK-LABEL: xmm_reg_i32:
// CHECK: #APP
// CHECK: movaps xmm{{[0-9]+}}, xmm{{[0-9]+}}
// CHECK: #NO_APP
check!(xmm_reg_i32 i32 xmm_reg "movaps");

// CHECK-LABEL: xmm_reg_f32:
// CHECK: #APP
// CHECK: movaps xmm{{[0-9]+}}, xmm{{[0-9]+}}
// CHECK: #NO_APP
check!(xmm_reg_f32 f32 xmm_reg "movaps");

// CHECK-LABEL: xmm_reg_i64:
// CHECK: #APP
// CHECK: movaps xmm{{[0-9]+}}, xmm{{[0-9]+}}
// CHECK: #NO_APP
check!(xmm_reg_i64 i64 xmm_reg "movaps");

// CHECK-LABEL: xmm_reg_f64:
// CHECK: #APP
// CHECK: movaps xmm{{[0-9]+}}, xmm{{[0-9]+}}
// CHECK: #NO_APP
check!(xmm_reg_f64 f64 xmm_reg "movaps");

// CHECK-LABEL: xmm_reg_f128:
// CHECK: #APP
// CHECK: movaps xmm{{[0-9]+}}, xmm{{[0-9]+}}
// CHECK: #NO_APP
check!(xmm_reg_f128 f128 xmm_reg "movaps");

// CHECK-LABEL: xmm_reg_ptr:
// CHECK: #APP
// CHECK: movaps xmm{{[0-9]+}}, xmm{{[0-9]+}}
// CHECK: #NO_APP
check!(xmm_reg_ptr ptr xmm_reg "movaps");

// CHECK-LABEL: xmm_reg_i8x16:
// CHECK: #APP
// CHECK: movaps xmm{{[0-9]+}}, xmm{{[0-9]+}}
// CHECK: #NO_APP
check!(xmm_reg_i8x16 i8x16 xmm_reg "movaps");

// CHECK-LABEL: xmm_reg_i16x8:
// CHECK: #APP
// CHECK: movaps xmm{{[0-9]+}}, xmm{{[0-9]+}}
// CHECK: #NO_APP
check!(xmm_reg_i16x8 i16x8 xmm_reg "movaps");

// CHECK-LABEL: xmm_reg_i32x4:
// CHECK: #APP
// CHECK: movaps xmm{{[0-9]+}}, xmm{{[0-9]+}}
// CHECK: #NO_APP
check!(xmm_reg_i32x4 i32x4 xmm_reg "movaps");

// CHECK-LABEL: xmm_reg_i64x2:
// CHECK: #APP
// CHECK: movaps xmm{{[0-9]+}}, xmm{{[0-9]+}}
// CHECK: #NO_APP
check!(xmm_reg_i64x2 i64x2 xmm_reg "movaps");

// CHECK-LABEL: xmm_reg_f16x8:
// CHECK: #APP
// CHECK: movaps xmm{{[0-9]+}}, xmm{{[0-9]+}}
// CHECK: #NO_APP
check!(xmm_reg_f16x8 f16x8 xmm_reg "movaps");

// CHECK-LABEL: xmm_reg_f32x4:
// CHECK: #APP
// CHECK: movaps xmm{{[0-9]+}}, xmm{{[0-9]+}}
// CHECK: #NO_APP
check!(xmm_reg_f32x4 f32x4 xmm_reg "movaps");

// CHECK-LABEL: xmm_reg_f64x2:
// CHECK: #APP
// CHECK: movaps xmm{{[0-9]+}}, xmm{{[0-9]+}}
// CHECK: #NO_APP
check!(xmm_reg_f64x2 f64x2 xmm_reg "movaps");

// CHECK-LABEL: ymm_reg_f16:
// CHECK: #APP
// CHECK: vmovaps ymm{{[0-9]+}}, ymm{{[0-9]+}}
// CHECK: #NO_APP
check!(ymm_reg_f16 f16 ymm_reg "vmovaps");

// CHECK-LABEL: ymm_reg_i32:
// CHECK: #APP
// CHECK: vmovaps ymm{{[0-9]+}}, ymm{{[0-9]+}}
// CHECK: #NO_APP
check!(ymm_reg_i32 i32 ymm_reg "vmovaps");

// CHECK-LABEL: ymm_reg_f32:
// CHECK: #APP
// CHECK: vmovaps ymm{{[0-9]+}}, ymm{{[0-9]+}}
// CHECK: #NO_APP
check!(ymm_reg_f32 f32 ymm_reg "vmovaps");

// CHECK-LABEL: ymm_reg_i64:
// CHECK: #APP
// CHECK: vmovaps ymm{{[0-9]+}}, ymm{{[0-9]+}}
// CHECK: #NO_APP
check!(ymm_reg_i64 i64 ymm_reg "vmovaps");

// CHECK-LABEL: ymm_reg_f64:
// CHECK: #APP
// CHECK: vmovaps ymm{{[0-9]+}}, ymm{{[0-9]+}}
// CHECK: #NO_APP
check!(ymm_reg_f64 f64 ymm_reg "vmovaps");

// CHECK-LABEL: ymm_reg_f128:
// CHECK: #APP
// CHECK: vmovaps ymm{{[0-9]+}}, ymm{{[0-9]+}}
// CHECK: #NO_APP
check!(ymm_reg_f128 f128 ymm_reg "vmovaps");

// CHECK-LABEL: ymm_reg_ptr:
// CHECK: #APP
// CHECK: vmovaps ymm{{[0-9]+}}, ymm{{[0-9]+}}
// CHECK: #NO_APP
check!(ymm_reg_ptr ptr ymm_reg "vmovaps");

// CHECK-LABEL: ymm_reg_i8x16:
// CHECK: #APP
// CHECK: vmovaps ymm{{[0-9]+}}, ymm{{[0-9]+}}
// CHECK: #NO_APP
check!(ymm_reg_i8x16 i8x16 ymm_reg "vmovaps");

// CHECK-LABEL: ymm_reg_i16x8:
// CHECK: #APP
// CHECK: vmovaps ymm{{[0-9]+}}, ymm{{[0-9]+}}
// CHECK: #NO_APP
check!(ymm_reg_i16x8 i16x8 ymm_reg "vmovaps");

// CHECK-LABEL: ymm_reg_i32x4:
// CHECK: #APP
// CHECK: vmovaps ymm{{[0-9]+}}, ymm{{[0-9]+}}
// CHECK: #NO_APP
check!(ymm_reg_i32x4 i32x4 ymm_reg "vmovaps");

// CHECK-LABEL: ymm_reg_i64x2:
// CHECK: #APP
// CHECK: vmovaps ymm{{[0-9]+}}, ymm{{[0-9]+}}
// CHECK: #NO_APP
check!(ymm_reg_i64x2 i64x2 ymm_reg "vmovaps");

// CHECK-LABEL: ymm_reg_f16x8:
// CHECK: #APP
// CHECK: vmovaps ymm{{[0-9]+}}, ymm{{[0-9]+}}
// CHECK: #NO_APP
check!(ymm_reg_f16x8 f16x8 ymm_reg "vmovaps");

// CHECK-LABEL: ymm_reg_f32x4:
// CHECK: #APP
// CHECK: vmovaps ymm{{[0-9]+}}, ymm{{[0-9]+}}
// CHECK: #NO_APP
check!(ymm_reg_f32x4 f32x4 ymm_reg "vmovaps");

// CHECK-LABEL: ymm_reg_f64x2:
// CHECK: #APP
// CHECK: vmovaps ymm{{[0-9]+}}, ymm{{[0-9]+}}
// CHECK: #NO_APP
check!(ymm_reg_f64x2 f64x2 ymm_reg "vmovaps");

// CHECK-LABEL: ymm_reg_i8x32:
// CHECK: #APP
// CHECK: vmovaps ymm{{[0-9]+}}, ymm{{[0-9]+}}
// CHECK: #NO_APP
check!(ymm_reg_i8x32 i8x32 ymm_reg "vmovaps");

// CHECK-LABEL: ymm_reg_i16x16:
// CHECK: #APP
// CHECK: vmovaps ymm{{[0-9]+}}, ymm{{[0-9]+}}
// CHECK: #NO_APP
check!(ymm_reg_i16x16 i16x16 ymm_reg "vmovaps");

// CHECK-LABEL: ymm_reg_i32x8:
// CHECK: #APP
// CHECK: vmovaps ymm{{[0-9]+}}, ymm{{[0-9]+}}
// CHECK: #NO_APP
check!(ymm_reg_i32x8 i32x8 ymm_reg "vmovaps");

// CHECK-LABEL: ymm_reg_i64x4:
// CHECK: #APP
// CHECK: vmovaps ymm{{[0-9]+}}, ymm{{[0-9]+}}
// CHECK: #NO_APP
check!(ymm_reg_i64x4 i64x4 ymm_reg "vmovaps");

// CHECK-LABEL: ymm_reg_f16x16:
// CHECK: #APP
// CHECK: vmovaps ymm{{[0-9]+}}, ymm{{[0-9]+}}
// CHECK: #NO_APP
check!(ymm_reg_f16x16 f16x16 ymm_reg "vmovaps");

// CHECK-LABEL: ymm_reg_f32x8:
// CHECK: #APP
// CHECK: vmovaps ymm{{[0-9]+}}, ymm{{[0-9]+}}
// CHECK: #NO_APP
check!(ymm_reg_f32x8 f32x8 ymm_reg "vmovaps");

// CHECK-LABEL: ymm_reg_f64x4:
// CHECK: #APP
// CHECK: vmovaps ymm{{[0-9]+}}, ymm{{[0-9]+}}
// CHECK: #NO_APP
check!(ymm_reg_f64x4 f64x4 ymm_reg "vmovaps");

// CHECK-LABEL: zmm_reg_f16:
// CHECK: #APP
// CHECK: vmovaps zmm{{[0-9]+}}, zmm{{[0-9]+}}
// CHECK: #NO_APP
check!(zmm_reg_f16 f16 zmm_reg "vmovaps");

// CHECK-LABEL: zmm_reg_i32:
// CHECK: #APP
// CHECK: vmovaps zmm{{[0-9]+}}, zmm{{[0-9]+}}
// CHECK: #NO_APP
check!(zmm_reg_i32 i32 zmm_reg "vmovaps");

// CHECK-LABEL: zmm_reg_f32:
// CHECK: #APP
// CHECK: vmovaps zmm{{[0-9]+}}, zmm{{[0-9]+}}
// CHECK: #NO_APP
check!(zmm_reg_f32 f32 zmm_reg "vmovaps");

// CHECK-LABEL: zmm_reg_i64:
// CHECK: #APP
// CHECK: vmovaps zmm{{[0-9]+}}, zmm{{[0-9]+}}
// CHECK: #NO_APP
check!(zmm_reg_i64 i64 zmm_reg "vmovaps");

// CHECK-LABEL: zmm_reg_f64:
// CHECK: #APP
// CHECK: vmovaps zmm{{[0-9]+}}, zmm{{[0-9]+}}
// CHECK: #NO_APP
check!(zmm_reg_f64 f64 zmm_reg "vmovaps");

// CHECK-LABEL: zmm_reg_f128:
// CHECK: #APP
// CHECK: vmovaps zmm{{[0-9]+}}, zmm{{[0-9]+}}
// CHECK: #NO_APP
check!(zmm_reg_f128 f128 zmm_reg "vmovaps");

// CHECK-LABEL: zmm_reg_ptr:
// CHECK: #APP
// CHECK: vmovaps zmm{{[0-9]+}}, zmm{{[0-9]+}}
// CHECK: #NO_APP
check!(zmm_reg_ptr ptr zmm_reg "vmovaps");

// CHECK-LABEL: zmm_reg_i8x16:
// CHECK: #APP
// CHECK: vmovaps zmm{{[0-9]+}}, zmm{{[0-9]+}}
// CHECK: #NO_APP
check!(zmm_reg_i8x16 i8x16 zmm_reg "vmovaps");

// CHECK-LABEL: zmm_reg_i16x8:
// CHECK: #APP
// CHECK: vmovaps zmm{{[0-9]+}}, zmm{{[0-9]+}}
// CHECK: #NO_APP
check!(zmm_reg_i16x8 i16x8 zmm_reg "vmovaps");

// CHECK-LABEL: zmm_reg_i32x4:
// CHECK: #APP
// CHECK: vmovaps zmm{{[0-9]+}}, zmm{{[0-9]+}}
// CHECK: #NO_APP
check!(zmm_reg_i32x4 i32x4 zmm_reg "vmovaps");

// CHECK-LABEL: zmm_reg_i64x2:
// CHECK: #APP
// CHECK: vmovaps zmm{{[0-9]+}}, zmm{{[0-9]+}}
// CHECK: #NO_APP
check!(zmm_reg_i64x2 i64x2 zmm_reg "vmovaps");

// CHECK-LABEL: zmm_reg_f16x8:
// CHECK: #APP
// CHECK: vmovaps zmm{{[0-9]+}}, zmm{{[0-9]+}}
// CHECK: #NO_APP
check!(zmm_reg_f16x8 f16x8 zmm_reg "vmovaps");

// CHECK-LABEL: zmm_reg_f32x4:
// CHECK: #APP
// CHECK: vmovaps zmm{{[0-9]+}}, zmm{{[0-9]+}}
// CHECK: #NO_APP
check!(zmm_reg_f32x4 f32x4 zmm_reg "vmovaps");

// CHECK-LABEL: zmm_reg_f64x2:
// CHECK: #APP
// CHECK: vmovaps zmm{{[0-9]+}}, zmm{{[0-9]+}}
// CHECK: #NO_APP
check!(zmm_reg_f64x2 f64x2 zmm_reg "vmovaps");

// CHECK-LABEL: zmm_reg_i8x32:
// CHECK: #APP
// CHECK: vmovaps zmm{{[0-9]+}}, zmm{{[0-9]+}}
// CHECK: #NO_APP
check!(zmm_reg_i8x32 i8x32 zmm_reg "vmovaps");

// CHECK-LABEL: zmm_reg_i16x16:
// CHECK: #APP
// CHECK: vmovaps zmm{{[0-9]+}}, zmm{{[0-9]+}}
// CHECK: #NO_APP
check!(zmm_reg_i16x16 i16x16 zmm_reg "vmovaps");

// CHECK-LABEL: zmm_reg_i32x8:
// CHECK: #APP
// CHECK: vmovaps zmm{{[0-9]+}}, zmm{{[0-9]+}}
// CHECK: #NO_APP
check!(zmm_reg_i32x8 i32x8 zmm_reg "vmovaps");

// CHECK-LABEL: zmm_reg_i64x4:
// CHECK: #APP
// CHECK: vmovaps zmm{{[0-9]+}}, zmm{{[0-9]+}}
// CHECK: #NO_APP
check!(zmm_reg_i64x4 i64x4 zmm_reg "vmovaps");

// CHECK-LABEL: zmm_reg_f16x16:
// CHECK: #APP
// CHECK: vmovaps zmm{{[0-9]+}}, zmm{{[0-9]+}}
// CHECK: #NO_APP
check!(zmm_reg_f16x16 f16x16 zmm_reg "vmovaps");

// CHECK-LABEL: zmm_reg_f32x8:
// CHECK: #APP
// CHECK: vmovaps zmm{{[0-9]+}}, zmm{{[0-9]+}}
// CHECK: #NO_APP
check!(zmm_reg_f32x8 f32x8 zmm_reg "vmovaps");

// CHECK-LABEL: zmm_reg_f64x4:
// CHECK: #APP
// CHECK: vmovaps zmm{{[0-9]+}}, zmm{{[0-9]+}}
// CHECK: #NO_APP
check!(zmm_reg_f64x4 f64x4 zmm_reg "vmovaps");

// CHECK-LABEL: zmm_reg_i8x64:
// CHECK: #APP
// CHECK: vmovaps zmm{{[0-9]+}}, zmm{{[0-9]+}}
// CHECK: #NO_APP
check!(zmm_reg_i8x64 i8x64 zmm_reg "vmovaps");

// CHECK-LABEL: zmm_reg_i16x32:
// CHECK: #APP
// CHECK: vmovaps zmm{{[0-9]+}}, zmm{{[0-9]+}}
// CHECK: #NO_APP
check!(zmm_reg_i16x32 i16x32 zmm_reg "vmovaps");

// CHECK-LABEL: zmm_reg_i32x16:
// CHECK: #APP
// CHECK: vmovaps zmm{{[0-9]+}}, zmm{{[0-9]+}}
// CHECK: #NO_APP
check!(zmm_reg_i32x16 i32x16 zmm_reg "vmovaps");

// CHECK-LABEL: zmm_reg_i64x8:
// CHECK: #APP
// CHECK: vmovaps zmm{{[0-9]+}}, zmm{{[0-9]+}}
// CHECK: #NO_APP
check!(zmm_reg_i64x8 i64x8 zmm_reg "vmovaps");

// CHECK-LABEL: zmm_reg_f16x32:
// CHECK: #APP
// CHECK: vmovaps zmm{{[0-9]+}}, zmm{{[0-9]+}}
// CHECK: #NO_APP
check!(zmm_reg_f16x32 f16x32 zmm_reg "vmovaps");

// CHECK-LABEL: zmm_reg_f32x16:
// CHECK: #APP
// CHECK: vmovaps zmm{{[0-9]+}}, zmm{{[0-9]+}}
// CHECK: #NO_APP
check!(zmm_reg_f32x16 f32x16 zmm_reg "vmovaps");

// CHECK-LABEL: zmm_reg_f64x8:
// CHECK: #APP
// CHECK: vmovaps zmm{{[0-9]+}}, zmm{{[0-9]+}}
// CHECK: #NO_APP
check!(zmm_reg_f64x8 f64x8 zmm_reg "vmovaps");

// CHECK-LABEL: kreg_i8:
// CHECK: #APP
// CHECK: kmovb k{{[0-9]+}}, k{{[0-9]+}}
// CHECK: #NO_APP
check!(kreg_i8 i8 kreg "kmovb");

// CHECK-LABEL: kreg_i16:
// CHECK: #APP
// CHECK: kmovw k{{[0-9]+}}, k{{[0-9]+}}
// CHECK: #NO_APP
check!(kreg_i16 i16 kreg "kmovw");

// CHECK-LABEL: kreg_i32:
// CHECK: #APP
// CHECK: kmovd k{{[0-9]+}}, k{{[0-9]+}}
// CHECK: #NO_APP
check!(kreg_i32 i32 kreg "kmovd");

// CHECK-LABEL: kreg_i64:
// CHECK: #APP
// CHECK: kmovq k{{[0-9]+}}, k{{[0-9]+}}
// CHECK: #NO_APP
check!(kreg_i64 i64 kreg "kmovq");

// CHECK-LABEL: kreg_ptr:
// CHECK: #APP
// CHECK: kmovq k{{[0-9]+}}, k{{[0-9]+}}
// CHECK: #NO_APP
check!(kreg_ptr ptr kreg "kmovq");

// CHECK-LABEL: eax_i16:
// CHECK: #APP
// CHECK: mov eax, eax
// CHECK: #NO_APP
check_reg!(eax_i16 i16 "eax" "mov");

// CHECK-LABEL: eax_f16:
// CHECK: #APP
// CHECK: mov eax, eax
// CHECK: #NO_APP
check_reg!(eax_f16 f16 "eax" "mov");

// CHECK-LABEL: eax_i32:
// CHECK: #APP
// CHECK: mov eax, eax
// CHECK: #NO_APP
check_reg!(eax_i32 i32 "eax" "mov");

// CHECK-LABEL: eax_f32:
// CHECK: #APP
// CHECK: mov eax, eax
// CHECK: #NO_APP
check_reg!(eax_f32 f32 "eax" "mov");

// x86_64-LABEL: eax_i64:
// x86_64: #APP
// x86_64: mov eax, eax
// x86_64: #NO_APP
#[cfg(x86_64)]
check_reg!(eax_i64 i64 "eax" "mov");

// x86_64-LABEL: eax_f64:
// x86_64: #APP
// x86_64: mov eax, eax
// x86_64: #NO_APP
#[cfg(x86_64)]
check_reg!(eax_f64 f64 "eax" "mov");

// CHECK-LABEL: eax_ptr:
// CHECK: #APP
// CHECK: mov eax, eax
// CHECK: #NO_APP
check_reg!(eax_ptr ptr "eax" "mov");

// i686-LABEL: ah_byte:
// i686: #APP
// i686: mov ah, ah
// i686: #NO_APP
#[cfg(i686)]
check_reg!(ah_byte i8 "ah" "mov");

// CHECK-LABEL: xmm0_f16:
// CHECK: #APP
// CHECK: movaps xmm0, xmm0
// CHECK: #NO_APP
check_reg!(xmm0_f16 f16 "xmm0" "movaps");

// CHECK-LABEL: xmm0_i32:
// CHECK: #APP
// CHECK: movaps xmm0, xmm0
// CHECK: #NO_APP
check_reg!(xmm0_i32 i32 "xmm0" "movaps");

// CHECK-LABEL: xmm0_f32:
// CHECK: #APP
// CHECK: movaps xmm0, xmm0
// CHECK: #NO_APP
check_reg!(xmm0_f32 f32 "xmm0" "movaps");

// CHECK-LABEL: xmm0_i64:
// CHECK: #APP
// CHECK: movaps xmm0, xmm0
// CHECK: #NO_APP
check_reg!(xmm0_i64 i64 "xmm0" "movaps");

// CHECK-LABEL: xmm0_f64:
// CHECK: #APP
// CHECK: movaps xmm0, xmm0
// CHECK: #NO_APP
check_reg!(xmm0_f64 f64 "xmm0" "movaps");

// CHECK-LABEL: xmm0_f128:
// CHECK: #APP
// CHECK: movaps xmm0, xmm0
// CHECK: #NO_APP
check_reg!(xmm0_f128 f128 "xmm0" "movaps");

// CHECK-LABEL: xmm0_ptr:
// CHECK: #APP
// CHECK: movaps xmm0, xmm0
// CHECK: #NO_APP
check_reg!(xmm0_ptr ptr "xmm0" "movaps");

// CHECK-LABEL: xmm0_i8x16:
// CHECK: #APP
// CHECK: movaps xmm0, xmm0
// CHECK: #NO_APP
check_reg!(xmm0_i8x16 i8x16 "xmm0" "movaps");

// CHECK-LABEL: xmm0_i16x8:
// CHECK: #APP
// CHECK: movaps xmm0, xmm0
// CHECK: #NO_APP
check_reg!(xmm0_i16x8 i16x8 "xmm0" "movaps");

// CHECK-LABEL: xmm0_i32x4:
// CHECK: #APP
// CHECK: movaps xmm0, xmm0
// CHECK: #NO_APP
check_reg!(xmm0_i32x4 i32x4 "xmm0" "movaps");

// CHECK-LABEL: xmm0_i64x2:
// CHECK: #APP
// CHECK: movaps xmm0, xmm0
// CHECK: #NO_APP
check_reg!(xmm0_i64x2 i64x2 "xmm0" "movaps");

// CHECK-LABEL: xmm0_f16x8:
// CHECK: #APP
// CHECK: movaps xmm0, xmm0
// CHECK: #NO_APP
check_reg!(xmm0_f16x8 f16x8 "xmm0" "movaps");

// CHECK-LABEL: xmm0_f32x4:
// CHECK: #APP
// CHECK: movaps xmm0, xmm0
// CHECK: #NO_APP
check_reg!(xmm0_f32x4 f32x4 "xmm0" "movaps");

// CHECK-LABEL: xmm0_f64x2:
// CHECK: #APP
// CHECK: movaps xmm0, xmm0
// CHECK: #NO_APP
check_reg!(xmm0_f64x2 f64x2 "xmm0" "movaps");

// CHECK-LABEL: ymm0_f16:
// CHECK: #APP
// CHECK: vmovaps ymm0, ymm0
// CHECK: #NO_APP
check_reg!(ymm0_f16 f16 "ymm0" "vmovaps");

// CHECK-LABEL: ymm0_i32:
// CHECK: #APP
// CHECK: vmovaps ymm0, ymm0
// CHECK: #NO_APP
check_reg!(ymm0_i32 i32 "ymm0" "vmovaps");

// CHECK-LABEL: ymm0_f32:
// CHECK: #APP
// CHECK: vmovaps ymm0, ymm0
// CHECK: #NO_APP
check_reg!(ymm0_f32 f32 "ymm0" "vmovaps");

// CHECK-LABEL: ymm0_i64:
// CHECK: #APP
// CHECK: vmovaps ymm0, ymm0
// CHECK: #NO_APP
check_reg!(ymm0_i64 i64 "ymm0" "vmovaps");

// CHECK-LABEL: ymm0_f64:
// CHECK: #APP
// CHECK: vmovaps ymm0, ymm0
// CHECK: #NO_APP
check_reg!(ymm0_f64 f64 "ymm0" "vmovaps");

// CHECK-LABEL: ymm0_f128:
// CHECK: #APP
// CHECK: vmovaps ymm0, ymm0
// CHECK: #NO_APP
check_reg!(ymm0_f128 f128 "ymm0" "vmovaps");

// CHECK-LABEL: ymm0_ptr:
// CHECK: #APP
// CHECK: vmovaps ymm0, ymm0
// CHECK: #NO_APP
check_reg!(ymm0_ptr ptr "ymm0" "vmovaps");

// CHECK-LABEL: ymm0_i8x16:
// CHECK: #APP
// CHECK: vmovaps ymm0, ymm0
// CHECK: #NO_APP
check_reg!(ymm0_i8x16 i8x16 "ymm0" "vmovaps");

// CHECK-LABEL: ymm0_i16x8:
// CHECK: #APP
// CHECK: vmovaps ymm0, ymm0
// CHECK: #NO_APP
check_reg!(ymm0_i16x8 i16x8 "ymm0" "vmovaps");

// CHECK-LABEL: ymm0_i32x4:
// CHECK: #APP
// CHECK: vmovaps ymm0, ymm0
// CHECK: #NO_APP
check_reg!(ymm0_i32x4 i32x4 "ymm0" "vmovaps");

// CHECK-LABEL: ymm0_i64x2:
// CHECK: #APP
// CHECK: vmovaps ymm0, ymm0
// CHECK: #NO_APP
check_reg!(ymm0_i64x2 i64x2 "ymm0" "vmovaps");

// CHECK-LABEL: ymm0_f16x8:
// CHECK: #APP
// CHECK: vmovaps ymm0, ymm0
// CHECK: #NO_APP
check_reg!(ymm0_f16x8 f16x8 "ymm0" "vmovaps");

// CHECK-LABEL: ymm0_f32x4:
// CHECK: #APP
// CHECK: vmovaps ymm0, ymm0
// CHECK: #NO_APP
check_reg!(ymm0_f32x4 f32x4 "ymm0" "vmovaps");

// CHECK-LABEL: ymm0_f64x2:
// CHECK: #APP
// CHECK: vmovaps ymm0, ymm0
// CHECK: #NO_APP
check_reg!(ymm0_f64x2 f64x2 "ymm0" "vmovaps");

// CHECK-LABEL: ymm0_i8x32:
// CHECK: #APP
// CHECK: vmovaps ymm0, ymm0
// CHECK: #NO_APP
check_reg!(ymm0_i8x32 i8x32 "ymm0" "vmovaps");

// CHECK-LABEL: ymm0_i16x16:
// CHECK: #APP
// CHECK: vmovaps ymm0, ymm0
// CHECK: #NO_APP
check_reg!(ymm0_i16x16 i16x16 "ymm0" "vmovaps");

// CHECK-LABEL: ymm0_i32x8:
// CHECK: #APP
// CHECK: vmovaps ymm0, ymm0
// CHECK: #NO_APP
check_reg!(ymm0_i32x8 i32x8 "ymm0" "vmovaps");

// CHECK-LABEL: ymm0_i64x4:
// CHECK: #APP
// CHECK: vmovaps ymm0, ymm0
// CHECK: #NO_APP
check_reg!(ymm0_i64x4 i64x4 "ymm0" "vmovaps");

// CHECK-LABEL: ymm0_f16x16:
// CHECK: #APP
// CHECK: vmovaps ymm0, ymm0
// CHECK: #NO_APP
check_reg!(ymm0_f16x16 f16x16 "ymm0" "vmovaps");

// CHECK-LABEL: ymm0_f32x8:
// CHECK: #APP
// CHECK: vmovaps ymm0, ymm0
// CHECK: #NO_APP
check_reg!(ymm0_f32x8 f32x8 "ymm0" "vmovaps");

// CHECK-LABEL: ymm0_f64x4:
// CHECK: #APP
// CHECK: vmovaps ymm0, ymm0
// CHECK: #NO_APP
check_reg!(ymm0_f64x4 f64x4 "ymm0" "vmovaps");

// CHECK-LABEL: zmm0_f16:
// CHECK: #APP
// CHECK: vmovaps zmm0, zmm0
// CHECK: #NO_APP
check_reg!(zmm0_f16 f16 "zmm0" "vmovaps");

// CHECK-LABEL: zmm0_i32:
// CHECK: #APP
// CHECK: vmovaps zmm0, zmm0
// CHECK: #NO_APP
check_reg!(zmm0_i32 i32 "zmm0" "vmovaps");

// CHECK-LABEL: zmm0_f32:
// CHECK: #APP
// CHECK: vmovaps zmm0, zmm0
// CHECK: #NO_APP
check_reg!(zmm0_f32 f32 "zmm0" "vmovaps");

// CHECK-LABEL: zmm0_i64:
// CHECK: #APP
// CHECK: vmovaps zmm0, zmm0
// CHECK: #NO_APP
check_reg!(zmm0_i64 i64 "zmm0" "vmovaps");

// CHECK-LABEL: zmm0_f64:
// CHECK: #APP
// CHECK: vmovaps zmm0, zmm0
// CHECK: #NO_APP
check_reg!(zmm0_f64 f64 "zmm0" "vmovaps");

// CHECK-LABEL: zmm0_f128:
// CHECK: #APP
// CHECK: vmovaps zmm0, zmm0
// CHECK: #NO_APP
check_reg!(zmm0_f128 f128 "zmm0" "vmovaps");

// CHECK-LABEL: zmm0_ptr:
// CHECK: #APP
// CHECK: vmovaps zmm0, zmm0
// CHECK: #NO_APP
check_reg!(zmm0_ptr ptr "zmm0" "vmovaps");

// CHECK-LABEL: zmm0_i8x16:
// CHECK: #APP
// CHECK: vmovaps zmm0, zmm0
// CHECK: #NO_APP
check_reg!(zmm0_i8x16 i8x16 "zmm0" "vmovaps");

// CHECK-LABEL: zmm0_i16x8:
// CHECK: #APP
// CHECK: vmovaps zmm0, zmm0
// CHECK: #NO_APP
check_reg!(zmm0_i16x8 i16x8 "zmm0" "vmovaps");

// CHECK-LABEL: zmm0_i32x4:
// CHECK: #APP
// CHECK: vmovaps zmm0, zmm0
// CHECK: #NO_APP
check_reg!(zmm0_i32x4 i32x4 "zmm0" "vmovaps");

// CHECK-LABEL: zmm0_i64x2:
// CHECK: #APP
// CHECK: vmovaps zmm0, zmm0
// CHECK: #NO_APP
check_reg!(zmm0_i64x2 i64x2 "zmm0" "vmovaps");

// CHECK-LABEL: zmm0_f16x8:
// CHECK: #APP
// CHECK: vmovaps zmm0, zmm0
// CHECK: #NO_APP
check_reg!(zmm0_f16x8 f16x8 "zmm0" "vmovaps");

// CHECK-LABEL: zmm0_f32x4:
// CHECK: #APP
// CHECK: vmovaps zmm0, zmm0
// CHECK: #NO_APP
check_reg!(zmm0_f32x4 f32x4 "zmm0" "vmovaps");

// CHECK-LABEL: zmm0_f64x2:
// CHECK: #APP
// CHECK: vmovaps zmm0, zmm0
// CHECK: #NO_APP
check_reg!(zmm0_f64x2 f64x2 "zmm0" "vmovaps");

// CHECK-LABEL: zmm0_i8x32:
// CHECK: #APP
// CHECK: vmovaps zmm0, zmm0
// CHECK: #NO_APP
check_reg!(zmm0_i8x32 i8x32 "zmm0" "vmovaps");

// CHECK-LABEL: zmm0_i16x16:
// CHECK: #APP
// CHECK: vmovaps zmm0, zmm0
// CHECK: #NO_APP
check_reg!(zmm0_i16x16 i16x16 "zmm0" "vmovaps");

// CHECK-LABEL: zmm0_i32x8:
// CHECK: #APP
// CHECK: vmovaps zmm0, zmm0
// CHECK: #NO_APP
check_reg!(zmm0_i32x8 i32x8 "zmm0" "vmovaps");

// CHECK-LABEL: zmm0_i64x4:
// CHECK: #APP
// CHECK: vmovaps zmm0, zmm0
// CHECK: #NO_APP
check_reg!(zmm0_i64x4 i64x4 "zmm0" "vmovaps");

// CHECK-LABEL: zmm0_f16x16:
// CHECK: #APP
// CHECK: vmovaps zmm0, zmm0
// CHECK: #NO_APP
check_reg!(zmm0_f16x16 f16x16 "zmm0" "vmovaps");

// CHECK-LABEL: zmm0_f32x8:
// CHECK: #APP
// CHECK: vmovaps zmm0, zmm0
// CHECK: #NO_APP
check_reg!(zmm0_f32x8 f32x8 "zmm0" "vmovaps");

// CHECK-LABEL: zmm0_f64x4:
// CHECK: #APP
// CHECK: vmovaps zmm0, zmm0
// CHECK: #NO_APP
check_reg!(zmm0_f64x4 f64x4 "zmm0" "vmovaps");

// CHECK-LABEL: zmm0_i8x64:
// CHECK: #APP
// CHECK: vmovaps zmm0, zmm0
// CHECK: #NO_APP
check_reg!(zmm0_i8x64 i8x64 "zmm0" "vmovaps");

// CHECK-LABEL: zmm0_i16x32:
// CHECK: #APP
// CHECK: vmovaps zmm0, zmm0
// CHECK: #NO_APP
check_reg!(zmm0_i16x32 i16x32 "zmm0" "vmovaps");

// CHECK-LABEL: zmm0_i32x16:
// CHECK: #APP
// CHECK: vmovaps zmm0, zmm0
// CHECK: #NO_APP
check_reg!(zmm0_i32x16 i32x16 "zmm0" "vmovaps");

// CHECK-LABEL: zmm0_i64x8:
// CHECK: #APP
// CHECK: vmovaps zmm0, zmm0
// CHECK: #NO_APP
check_reg!(zmm0_i64x8 i64x8 "zmm0" "vmovaps");

// CHECK-LABEL: zmm0_f16x32:
// CHECK: #APP
// CHECK: vmovaps zmm0, zmm0
// CHECK: #NO_APP
check_reg!(zmm0_f16x32 f16x32 "zmm0" "vmovaps");

// CHECK-LABEL: zmm0_f32x16:
// CHECK: #APP
// CHECK: vmovaps zmm0, zmm0
// CHECK: #NO_APP
check_reg!(zmm0_f32x16 f32x16 "zmm0" "vmovaps");

// CHECK-LABEL: zmm0_f64x8:
// CHECK: #APP
// CHECK: vmovaps zmm0, zmm0
// CHECK: #NO_APP
check_reg!(zmm0_f64x8 f64x8 "zmm0" "vmovaps");

// CHECK-LABEL: k1_i8:
// CHECK: #APP
// CHECK: kmovb k1, k1
// CHECK: #NO_APP
check_reg!(k1_i8 i8 "k1" "kmovb");

// CHECK-LABEL: k1_i16:
// CHECK: #APP
// CHECK: kmovw k1, k1
// CHECK: #NO_APP
check_reg!(k1_i16 i16 "k1" "kmovw");

// CHECK-LABEL: k1_i32:
// CHECK: #APP
// CHECK: kmovd k1, k1
// CHECK: #NO_APP
check_reg!(k1_i32 i32 "k1" "kmovd");

// CHECK-LABEL: k1_i64:
// CHECK: #APP
// CHECK: kmovq k1, k1
// CHECK: #NO_APP
check_reg!(k1_i64 i64 "k1" "kmovq");

// CHECK-LABEL: k1_ptr:
// CHECK: #APP
// CHECK: kmovq k1, k1
// CHECK: #NO_APP
check_reg!(k1_ptr ptr "k1" "kmovq");
