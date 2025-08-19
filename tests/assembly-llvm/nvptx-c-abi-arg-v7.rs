//@ assembly-output: ptx-linker
//@ compile-flags: --crate-type cdylib -C target-cpu=sm_86 -Z unstable-options -Clinker-flavor=llbc
//@ only-nvptx64

// The PTX ABI stability is tied to major versions of the PTX ISA
// These tests assume major version 7

// CHECK: .version 7

#![feature(abi_ptx, lang_items, no_core)]
#![no_core]

#[lang = "pointee_sized"]
trait PointeeSized {}
#[lang = "meta_sized"]
trait MetaSized: PointeeSized {}
#[lang = "sized"]
trait Sized: MetaSized {}
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
pub struct DoubleI32 {
    f: i32,
    g: i32,
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

// CHECK: .visible .func f_double_i32_arg(
// CHECK: .param .align 4 .b8 f_double_i32_arg_param_0[8]
#[no_mangle]
pub unsafe extern "C" fn f_double_i32_arg(_a: DoubleI32) {}

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
