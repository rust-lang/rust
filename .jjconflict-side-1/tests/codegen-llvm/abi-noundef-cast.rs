// Verify that `PassMode::Cast` arguments/returns in the Rust ABI carry `noundef`
// when the original layout provably contains no uninit bytes, and correctly omit
// it when uninit bytes or padding may be present.
//
// See <https://github.com/rust-lang/rust/issues/123183>.

//@ compile-flags: -Copt-level=3 -C no-prepopulate-passes
//@ only-64bit

#![crate_type = "lib"]

use std::mem::MaybeUninit;

// CHECK-LABEL: @arg_array_u32x2(
// CHECK-SAME: i64 noundef
#[no_mangle]
pub fn arg_array_u32x2(v: [u32; 2]) -> u32 {
    v[0]
}

// CHECK-LABEL: @arg_array_u8x4(
// CHECK-SAME: i32 noundef
#[no_mangle]
pub fn arg_array_u8x4(v: [u8; 4]) -> u8 {
    v[0]
}

// CHECK-LABEL: @arg_nested_array(
// CHECK-SAME: i64 noundef
#[no_mangle]
pub fn arg_nested_array(v: [[u8; 2]; 4]) -> u8 {
    v[0][0]
}

// CHECK-LABEL: @arg_array_bool(
// CHECK-SAME: i64 noundef
#[no_mangle]
pub fn arg_array_bool(v: [bool; 8]) -> bool {
    v[0]
}

struct FourU8 {
    a: u8,
    b: u8,
    c: u8,
    d: u8,
}

// CHECK-LABEL: @arg_four_u8(
// CHECK-SAME: i32 noundef
#[no_mangle]
pub fn arg_four_u8(v: FourU8) -> u8 {
    v.a
}

struct Wrapper([u32; 2]);

// CHECK-LABEL: @arg_newtype_wrapper(
// CHECK-SAME: i64 noundef
#[no_mangle]
pub fn arg_newtype_wrapper(v: Wrapper) -> u32 {
    (v.0)[0]
}

enum SingleVariant {
    Only([u32; 2]),
}

// CHECK-LABEL: @arg_single_variant_enum(
// CHECK-SAME: i64 noundef
#[no_mangle]
pub fn arg_single_variant_enum(v: SingleVariant) -> u32 {
    match v {
        SingleVariant::Only(a) => a[0],
    }
}

struct ContainsScalarPair {
    a: (u16, u16),
    b: u32,
}

// CHECK-LABEL: @arg_contains_scalar_pair(
// CHECK-SAME: i64 noundef
#[no_mangle]
pub fn arg_contains_scalar_pair(v: ContainsScalarPair) -> u32 {
    v.b
}

// CHECK: define noundef i64 @ret_array_u32x2(
#[no_mangle]
pub fn ret_array_u32x2(x: u32, y: u32) -> [u32; 2] {
    [x, y]
}

// CHECK-LABEL: @arg_maybeuninit_u8x8(
// CHECK-SAME: i64 %
#[no_mangle]
pub fn arg_maybeuninit_u8x8(v: [MaybeUninit<u8>; 8]) -> MaybeUninit<u8> {
    v[0]
}

enum MultiVariant {
    A(u8),
    B(u16),
    C,
}

// CHECK-LABEL: @arg_multi_variant_enum(
// CHECK-SAME: i32 %
#[no_mangle]
pub fn arg_multi_variant_enum(v: MultiVariant) -> u8 {
    match v {
        MultiVariant::A(x) => x,
        MultiVariant::B(_) | MultiVariant::C => 0,
    }
}

#[repr(C)]
struct HasFieldGap {
    a: u8,
    b: u16,
    c: u8,
}

// CHECK-LABEL: @arg_struct_field_gap(
// CHECK-SAME: i48 %
#[no_mangle]
pub fn arg_struct_field_gap(v: HasFieldGap) -> u8 {
    v.a
}

#[repr(C)]
struct HasPaddedPairField {
    a: (u8, u16),
    b: u8,
}

// CHECK-LABEL: @arg_struct_padded_pair_field(
// CHECK-SAME: i48 %
#[no_mangle]
pub fn arg_struct_padded_pair_field(v: HasPaddedPairField) -> u8 {
    v.b
}

#[repr(C)]
struct HasUndefPairField {
    a: (MaybeUninit<u16>, u16),
    b: u32,
}

// CHECK-LABEL: @arg_struct_undef_pair_field(
// CHECK-SAME: i64 %
#[no_mangle]
pub fn arg_struct_undef_pair_field(v: HasUndefPairField) -> u32 {
    v.b
}

// CHECK-LABEL: @arg_triple_maybeuninit_u8(
// CHECK-SAME: i24 %
#[no_mangle]
pub fn arg_triple_maybeuninit_u8(
    v: (MaybeUninit<u8>, MaybeUninit<u8>, MaybeUninit<u8>),
) -> MaybeUninit<u8> {
    v.0
}

#[repr(C)]
struct HasTrailingPadding {
    x: u32,
    y: u16,
    z: u8,
}

// CHECK-LABEL: @arg_struct_trailing_pad(
// CHECK-SAME: i64 %
#[no_mangle]
pub fn arg_struct_trailing_pad(v: HasTrailingPadding) -> u32 {
    v.x
}

// CHECK-LABEL: @arg_tuple_i8_i16(
// CHECK-SAME: i8 noundef
// CHECK-SAME: i16 noundef
#[no_mangle]
pub fn arg_tuple_i8_i16(v: (i8, i16)) -> i8 {
    v.0
}

// CHECK-LABEL: @arg_tuple_i16_maybeuninit(
// CHECK-SAME: i16 noundef
// CHECK-SAME: i16 %
#[no_mangle]
pub fn arg_tuple_i16_maybeuninit(v: (i16, MaybeUninit<i16>)) -> i16 {
    v.0
}

// CHECK-LABEL: @arg_result_i32(
// CHECK-SAME: i32 noundef
// CHECK-SAME: i32 noundef
#[no_mangle]
pub fn arg_result_i32(v: Result<i32, i32>) -> i32 {
    match v {
        Ok(x) | Err(x) => x,
    }
}
