// assembly-output: ptx-linker
// compile-flags: --crate-type cdylib -C target-cpu=sm_86
// only-nvptx64
// ignore-nvptx64

// The PTX ABI stability is tied to major versions of the PTX ISA
// These tests assume major version 7

// CHECK: .version 7

#![feature(abi_ptx, lang_items, no_core)]
#![no_core]

#[lang = "sized"]
trait Sized {}
#[lang = "copy"]
trait Copy {}

#[repr(C)]
pub struct SingleU8 {
    f: u8,
}

#[repr(C)]
pub struct DoubleU8 {
    f: u8,
    g: u8,
}

#[repr(C)]
pub struct TripleU8 {
    f: u8,
    g: u8,
    h: u8,
}

#[repr(C)]
pub struct TripleU16 {
    f: u16,
    g: u16,
    h: u16,
}
#[repr(C)]
pub struct TripleU32 {
    f: u32,
    g: u32,
    h: u32,
}
#[repr(C)]
pub struct TripleU64 {
    f: u64,
    g: u64,
    h: u64,
}

#[repr(C)]
pub struct DoubleFloat {
    f: f32,
    g: f32,
}

#[repr(C)]
pub struct TripleFloat {
    f: f32,
    g: f32,
    h: f32,
}

#[repr(C)]
pub struct TripleDouble {
    f: f64,
    g: f64,
    h: f64,
}

#[repr(C)]
pub struct ManyIntegers {
    f: u8,
    g: u16,
    h: u32,
    i: u64,
}

#[repr(C)]
pub struct ManyNumerics {
    f: u8,
    g: u16,
    h: u32,
    i: u64,
    j: f32,
    k: f64,
}

// CHECK: .visible .func f_u8_arg(
// CHECK: .param .b32 f_u8_arg_param_0
#[no_mangle]
pub unsafe extern "C" fn f_u8_arg(_a: u8) {}

// CHECK: .visible .func f_u16_arg(
// CHECK: .param .b32 f_u16_arg_param_0
#[no_mangle]
pub unsafe extern "C" fn f_u16_arg(_a: u16) {}

// CHECK: .visible .func f_u32_arg(
// CHECK: .param .b32 f_u32_arg_param_0
#[no_mangle]
pub unsafe extern "C" fn f_u32_arg(_a: u32) {}

// CHECK: .visible .func f_u64_arg(
// CHECK: .param .b64 f_u64_arg_param_0
#[no_mangle]
pub unsafe extern "C" fn f_u64_arg(_a: u64) {}

// CHECK: .visible .func f_u128_arg(
// CHECK: .param .align 16 .b8 f_u128_arg_param_0[16]
#[no_mangle]
pub unsafe extern "C" fn f_u128_arg(_a: u128) {}

// CHECK: .visible .func f_i8_arg(
// CHECK: .param .b32 f_i8_arg_param_0
#[no_mangle]
pub unsafe extern "C" fn f_i8_arg(_a: i8) {}

// CHECK: .visible .func f_i16_arg(
// CHECK: .param .b32 f_i16_arg_param_0
#[no_mangle]
pub unsafe extern "C" fn f_i16_arg(_a: i16) {}

// CHECK: .visible .func f_i32_arg(
// CHECK: .param .b32 f_i32_arg_param_0
#[no_mangle]
pub unsafe extern "C" fn f_i32_arg(_a: i32) {}

// CHECK: .visible .func f_i64_arg(
// CHECK: .param .b64 f_i64_arg_param_0
#[no_mangle]
pub unsafe extern "C" fn f_i64_arg(_a: i64) {}

// CHECK: .visible .func f_i128_arg(
// CHECK: .param .align 16 .b8 f_i128_arg_param_0[16]
#[no_mangle]
pub unsafe extern "C" fn f_i128_arg(_a: i128) {}

// CHECK: .visible .func f_f32_arg(
// CHECK: .param .b32 f_f32_arg_param_0
#[no_mangle]
pub unsafe extern "C" fn f_f32_arg(_a: f32) {}

// CHECK: .visible .func f_f64_arg(
// CHECK: .param .b64 f_f64_arg_param_0
#[no_mangle]
pub unsafe extern "C" fn f_f64_arg(_a: f64) {}

// CHECK: .visible .func f_single_u8_arg(
// CHECK: .param .align 1 .b8 f_single_u8_arg_param_0[1]
#[no_mangle]
pub unsafe extern "C" fn f_single_u8_arg(_a: SingleU8) {}

// CHECK: .visible .func f_double_u8_arg(
// CHECK: .param .align 1 .b8 f_double_u8_arg_param_0[2]
#[no_mangle]
pub unsafe extern "C" fn f_double_u8_arg(_a: DoubleU8) {}

// CHECK: .visible .func f_triple_u8_arg(
// CHECK: .param .align 1 .b8 f_triple_u8_arg_param_0[3]
#[no_mangle]
pub unsafe extern "C" fn f_triple_u8_arg(_a: TripleU8) {}

// CHECK: .visible .func f_triple_u16_arg(
// CHECK: .param .align 2 .b8 f_triple_u16_arg_param_0[6]
#[no_mangle]
pub unsafe extern "C" fn f_triple_u16_arg(_a: TripleU16) {}

// CHECK: .visible .func f_triple_u32_arg(
// CHECK: .param .align 4 .b8 f_triple_u32_arg_param_0[12]
#[no_mangle]
pub unsafe extern "C" fn f_triple_u32_arg(_a: TripleU32) {}

// CHECK: .visible .func f_triple_u64_arg(
// CHECK: .param .align 8 .b8 f_triple_u64_arg_param_0[24]
#[no_mangle]
pub unsafe extern "C" fn f_triple_u64_arg(_a: TripleU64) {}

// CHECK: .visible .func f_many_integers_arg(
// CHECK: .param .align 8 .b8 f_many_integers_arg_param_0[16]
#[no_mangle]
pub unsafe extern "C" fn f_many_integers_arg(_a: ManyIntegers) {}

// CHECK: .visible .func f_double_float_arg(
// CHECK: .param .align 4 .b8 f_double_float_arg_param_0[8]
#[no_mangle]
pub unsafe extern "C" fn f_double_float_arg(_a: DoubleFloat) {}

// CHECK: .visible .func f_triple_float_arg(
// CHECK: .param .align 4 .b8 f_triple_float_arg_param_0[12]
#[no_mangle]
pub unsafe extern "C" fn f_triple_float_arg(_a: TripleFloat) {}

// CHECK: .visible .func f_triple_double_arg(
// CHECK: .param .align 8 .b8 f_triple_double_arg_param_0[24]
#[no_mangle]
pub unsafe extern "C" fn f_triple_double_arg(_a: TripleDouble) {}

// CHECK: .visible .func f_many_numerics_arg(
// CHECK: .param .align 8 .b8 f_many_numerics_arg_param_0[32]
#[no_mangle]
pub unsafe extern "C" fn f_many_numerics_arg(_a: ManyNumerics) {}

// CHECK: .visible .func (.param .b32 func_retval0) f_u8_ret(
#[no_mangle]
pub unsafe extern "C" fn f_u8_ret() -> u8{0}

// CHECK: .visible .func (.param .b32 func_retval0) f_u16_ret(
#[no_mangle]
pub unsafe extern "C" fn f_u16_ret() -> u16 {1}

// CHECK: .visible .func (.param .b32 func_retval0) f_u32_ret(
#[no_mangle]
pub unsafe extern "C" fn f_u32_ret() -> u32 {2}

// CHECK: .visible .func (.param .b64 func_retval0) f_u64_ret(
#[no_mangle]
pub unsafe extern "C" fn f_u64_ret() -> u64 {3}

// CHECK: .visible .func (.param .align 16 .b8 func_retval0[16]) f_u128_ret(
#[no_mangle]
pub unsafe extern "C" fn f_u128_ret() -> u128 {4}

// CHECK: .visible .func (.param .b32 func_retval0) f_i8_ret(
#[no_mangle]
pub unsafe extern "C" fn f_i8_ret() -> i8 {5}

// CHECK: .visible .func (.param .b32 func_retval0) f_i16_ret(
#[no_mangle]
pub unsafe extern "C" fn f_i16_ret() -> i16 {6}

// CHECK: .visible .func (.param .b32 func_retval0) f_i32_ret(
#[no_mangle]
pub unsafe extern "C" fn f_i32_ret() -> i32 {7}

// CHECK: .visible .func (.param .b64 func_retval0) f_i64_ret(
#[no_mangle]
pub unsafe extern "C" fn f_i64_ret() -> i64 {8}

// CHECK: .visible .func  (.param .align 16 .b8 func_retval0[16]) f_i128_ret(
#[no_mangle]
pub unsafe extern "C" fn f_i128_ret() -> i128 {9}

// CHECK: .visible .func (.param .b32 func_retval0) f_f32_ret(
#[no_mangle]
pub unsafe extern "C" fn f_f32_ret() -> f32 {10.0}

// CHECK: .visible .func (.param .b64 func_retval0) f_f64_ret(
#[no_mangle]
pub unsafe extern "C" fn f_f64_ret() -> f64 {11.0}

// CHECK: .visible .func (.param .align 1 .b8 func_retval0[1]) f_single_u8_ret(
#[no_mangle]
pub unsafe extern "C" fn f_single_u8_ret() -> SingleU8 {SingleU8{f: 12}}

// CHECK: .visible .func (.param .align 1 .b8 func_retval0[2]) f_double_u8_ret(
#[no_mangle]
pub unsafe extern "C" fn f_double_u8_ret() -> DoubleU8 {DoubleU8{f: 13, g: 14}}

// CHECK: .visible .func (.param .align 1 .b8 func_retval0[3]) f_triple_u8_ret(
#[no_mangle]
pub unsafe extern "C" fn f_triple_u8_ret() -> TripleU8 {TripleU8{f: 15, g: 16, h: 17}}

// CHECK: .visible .func (.param .align 2 .b8 func_retval0[6]) f_triple_u16_ret(
#[no_mangle]
pub unsafe extern "C" fn f_triple_u16_ret() -> TripleU16 {TripleU16{f: 18, g: 19, h: 20}}

// CHECK: .visible .func (.param .align 4 .b8 func_retval0[12]) f_triple_u32_ret(
#[no_mangle]
pub unsafe extern "C" fn f_triple_u32_ret() -> TripleU32{TripleU32{f: 20, g: 21, h: 22}}

// CHECK: .visible .func (.param .align 8 .b8 func_retval0[24]) f_triple_u64_ret(
#[no_mangle]
pub unsafe extern "C" fn f_triple_u64_ret() -> TripleU64 {TripleU64 {f: 23, g: 24, h: 25}}

// CHECK: .visible .func (.param .align 8 .b8 func_retval0[16]) f_many_integers_ret(
#[no_mangle]
pub unsafe extern "C" fn f_many_integers_ret() -> ManyIntegers {
    ManyIntegers{f: 26, g: 27, h: 28, i: 29}
}

// CHECK: .visible .func (.param .align 4 .b8 func_retval0[8]) f_double_float_ret(
#[no_mangle]
pub unsafe extern "C" fn f_double_float_ret() -> DoubleFloat {
    DoubleFloat{f: 29.0, g: 30.0}
}

// CHECK: .visible .func (.param .align 4 .b8 func_retval0[12]) f_triple_float_ret(
#[no_mangle]
pub unsafe extern "C" fn f_triple_float_ret() -> TripleFloat {
    TripleFloat{f: 31.0, g: 32.0, h: 33.0}
}

// CHECK: .visible .func (.param .align 8 .b8 func_retval0[24]) f_triple_double_ret(
#[no_mangle]
pub unsafe extern "C" fn f_triple_double_ret() -> TripleDouble {
    TripleDouble{f: 34.0, g: 35.0, h: 36.0}
}

// CHECK: .visible .func (.param .align 8 .b8 func_retval0[32]) f_many_numerics_ret(
#[no_mangle]
pub unsafe extern "C" fn f_many_numerics_ret() -> ManyNumerics {
    ManyNumerics{
        f: 37,
        g: 38,
        h: 39,
        i: 40,
        j: 41.0,
        k: 43.0,
    }
}
