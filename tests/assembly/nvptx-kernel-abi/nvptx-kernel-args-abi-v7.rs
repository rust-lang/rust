//@ assembly-output: ptx-linker
//@ compile-flags: --crate-type cdylib -C target-cpu=sm_86 -Z unstable-options -Clinker-flavor=llbc
//@ only-nvptx64

// The following ABI tests are made with nvcc 11.6 does.
//
// The PTX ABI stability is tied to major versions of the PTX ISA
// These tests assume major version 7
//
//
// The following correspondence between types are assumed:
// u<N> - uint<N>_t
// i<N> - int<N>_t
// [T, N] - std::array<T, N>
// &T - T const*
// &mut T - T*

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

// CHECK: .visible .entry f_u8_arg(
// CHECK: .param .u8 f_u8_arg_param_0
#[no_mangle]
pub unsafe extern "ptx-kernel" fn f_u8_arg(_a: u8) {}

// CHECK: .visible .entry f_u16_arg(
// CHECK: .param .u16 f_u16_arg_param_0
#[no_mangle]
pub unsafe extern "ptx-kernel" fn f_u16_arg(_a: u16) {}

// CHECK: .visible .entry f_u32_arg(
// CHECK: .param .u32 f_u32_arg_param_0
#[no_mangle]
pub unsafe extern "ptx-kernel" fn f_u32_arg(_a: u32) {}

// CHECK: .visible .entry f_u64_arg(
// CHECK: .param .u64 f_u64_arg_param_0
#[no_mangle]
pub unsafe extern "ptx-kernel" fn f_u64_arg(_a: u64) {}

// CHECK: .visible .entry f_u128_arg(
// CHECK: .param .align 16 .b8 f_u128_arg_param_0[16]
#[no_mangle]
pub unsafe extern "ptx-kernel" fn f_u128_arg(_a: u128) {}

// CHECK: .visible .entry f_i8_arg(
// CHECK: .param .u8 f_i8_arg_param_0
#[no_mangle]
pub unsafe extern "ptx-kernel" fn f_i8_arg(_a: i8) {}

// CHECK: .visible .entry f_i16_arg(
// CHECK: .param .u16 f_i16_arg_param_0
#[no_mangle]
pub unsafe extern "ptx-kernel" fn f_i16_arg(_a: i16) {}

// CHECK: .visible .entry f_i32_arg(
// CHECK: .param .u32 f_i32_arg_param_0
#[no_mangle]
pub unsafe extern "ptx-kernel" fn f_i32_arg(_a: i32) {}

// CHECK: .visible .entry f_i64_arg(
// CHECK: .param .u64 f_i64_arg_param_0
#[no_mangle]
pub unsafe extern "ptx-kernel" fn f_i64_arg(_a: i64) {}

// CHECK: .visible .entry f_i128_arg(
// CHECK: .param .align 16 .b8 f_i128_arg_param_0[16]
#[no_mangle]
pub unsafe extern "ptx-kernel" fn f_i128_arg(_a: i128) {}

// CHECK: .visible .entry f_f32_arg(
// CHECK: .param .f32 f_f32_arg_param_0
#[no_mangle]
pub unsafe extern "ptx-kernel" fn f_f32_arg(_a: f32) {}

// CHECK: .visible .entry f_f64_arg(
// CHECK: .param .f64 f_f64_arg_param_0
#[no_mangle]
pub unsafe extern "ptx-kernel" fn f_f64_arg(_a: f64) {}

// CHECK: .visible .entry f_single_u8_arg(
// CHECK: .param .align 1 .b8 f_single_u8_arg_param_0[1]
#[no_mangle]
pub unsafe extern "ptx-kernel" fn f_single_u8_arg(_a: SingleU8) {}

// CHECK: .visible .entry f_double_u8_arg(
// CHECK: .param .align 1 .b8 f_double_u8_arg_param_0[2]
#[no_mangle]
pub unsafe extern "ptx-kernel" fn f_double_u8_arg(_a: DoubleU8) {}

// CHECK: .visible .entry f_triple_u8_arg(
// CHECK: .param .align 1 .b8 f_triple_u8_arg_param_0[3]
#[no_mangle]
pub unsafe extern "ptx-kernel" fn f_triple_u8_arg(_a: TripleU8) {}

// CHECK: .visible .entry f_triple_u16_arg(
// CHECK: .param .align 2 .b8 f_triple_u16_arg_param_0[6]
#[no_mangle]
pub unsafe extern "ptx-kernel" fn f_triple_u16_arg(_a: TripleU16) {}

// CHECK: .visible .entry f_double_i32_arg(
// CHECK: .param .align 4 .b8 f_double_i32_arg_param_0[8]
#[no_mangle]
pub unsafe extern "ptx-kernel" fn f_double_i32_arg(_a: DoubleI32) {}

// CHECK: .visible .entry f_triple_u32_arg(
// CHECK: .param .align 4 .b8 f_triple_u32_arg_param_0[12]
#[no_mangle]
pub unsafe extern "ptx-kernel" fn f_triple_u32_arg(_a: TripleU32) {}

// CHECK: .visible .entry f_triple_u64_arg(
// CHECK: .param .align 8 .b8 f_triple_u64_arg_param_0[24]
#[no_mangle]
pub unsafe extern "ptx-kernel" fn f_triple_u64_arg(_a: TripleU64) {}

// CHECK: .visible .entry f_many_integers_arg(
// CHECK: .param .align 8 .b8 f_many_integers_arg_param_0[16]
#[no_mangle]
pub unsafe extern "ptx-kernel" fn f_many_integers_arg(_a: ManyIntegers) {}

// CHECK: .visible .entry f_double_float_arg(
// CHECK: .param .align 4 .b8 f_double_float_arg_param_0[8]
#[no_mangle]
pub unsafe extern "ptx-kernel" fn f_double_float_arg(_a: DoubleFloat) {}

// CHECK: .visible .entry f_triple_float_arg(
// CHECK: .param .align 4 .b8 f_triple_float_arg_param_0[12]
#[no_mangle]
pub unsafe extern "ptx-kernel" fn f_triple_float_arg(_a: TripleFloat) {}

// CHECK: .visible .entry f_triple_double_arg(
// CHECK: .param .align 8 .b8 f_triple_double_arg_param_0[24]
#[no_mangle]
pub unsafe extern "ptx-kernel" fn f_triple_double_arg(_a: TripleDouble) {}

// CHECK: .visible .entry f_many_numerics_arg(
// CHECK: .param .align 8 .b8 f_many_numerics_arg_param_0[32]
#[no_mangle]
pub unsafe extern "ptx-kernel" fn f_many_numerics_arg(_a: ManyNumerics) {}

// CHECK: .visible .entry f_byte_array_arg(
// CHECK: .param .align 1 .b8 f_byte_array_arg_param_0[5]
#[no_mangle]
pub unsafe extern "ptx-kernel" fn f_byte_array_arg(_a: [u8; 5]) {}

// CHECK: .visible .entry f_float_array_arg(
// CHECK: .param .align 4 .b8 f_float_array_arg_param_0[20]
#[no_mangle]
pub unsafe extern "ptx-kernel" fn f_float_array_arg(_a: [f32; 5]) {}

// FIXME: u128 started to break compilation with disabled CI
// NO_CHECK: .visible .entry f_u128_array_arg(
// NO_CHECK: .param .align 16 .b8 f_u128_array_arg_param_0[80]
//#[no_mangle]
//pub unsafe extern "ptx-kernel" fn f_u128_array_arg(_a: [u128; 5]) {}

// CHECK: .visible .entry f_u32_slice_arg(
// CHECK: .param .align 8 .b8 f_u32_slice_arg_param_0[16]
#[no_mangle]
pub unsafe extern "ptx-kernel" fn f_u32_slice_arg(_a: &[u32]) {}
