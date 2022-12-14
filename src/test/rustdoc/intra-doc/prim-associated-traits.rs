#![feature(never_type)]
use std::str::FromStr;

// @has 'prim_associated_traits/struct.Number.html' '//a[@href="{{channel}}/std/primitive.f64.html#method.from_str"]' 'f64::from_str()'
// @has 'prim_associated_traits/struct.Number.html' '//a[@href="{{channel}}/std/primitive.f32.html#method.from_str"]' 'f32::from_str()'
// @has 'prim_associated_traits/struct.Number.html' '//a[@href="{{channel}}/std/primitive.isize.html#method.from_str"]' 'isize::from_str()'
// @has 'prim_associated_traits/struct.Number.html' '//a[@href="{{channel}}/std/primitive.i8.html#method.from_str"]' 'i8::from_str()'
// @has 'prim_associated_traits/struct.Number.html' '//a[@href="{{channel}}/std/primitive.i16.html#method.from_str"]' 'i16::from_str()'
// @has 'prim_associated_traits/struct.Number.html' '//a[@href="{{channel}}/std/primitive.i32.html#method.from_str"]' 'i32::from_str()'
// @has 'prim_associated_traits/struct.Number.html' '//a[@href="{{channel}}/std/primitive.i64.html#method.from_str"]' 'i64::from_str()'
// @has 'prim_associated_traits/struct.Number.html' '//a[@href="{{channel}}/std/primitive.i128.html#method.from_str"]' 'i128::from_str()'
// @has 'prim_associated_traits/struct.Number.html' '//a[@href="{{channel}}/std/primitive.usize.html#method.from_str"]' 'usize::from_str()'
// @has 'prim_associated_traits/struct.Number.html' '//a[@href="{{channel}}/std/primitive.u8.html#method.from_str"]' 'u8::from_str()'
// @has 'prim_associated_traits/struct.Number.html' '//a[@href="{{channel}}/std/primitive.u16.html#method.from_str"]' 'u16::from_str()'
// @has 'prim_associated_traits/struct.Number.html' '//a[@href="{{channel}}/std/primitive.u32.html#method.from_str"]' 'u32::from_str()'
// @has 'prim_associated_traits/struct.Number.html' '//a[@href="{{channel}}/std/primitive.u64.html#method.from_str"]' 'u64::from_str()'
// @has 'prim_associated_traits/struct.Number.html' '//a[@href="{{channel}}/std/primitive.u128.html#method.from_str"]' 'u128::from_str()'
// @has 'prim_associated_traits/struct.Number.html' '//a[@href="{{channel}}/std/primitive.char.html#method.from_str"]' 'char::from_str()'
// @has 'prim_associated_traits/struct.Number.html' '//a[@href="{{channel}}/std/primitive.bool.html#method.from_str"]' 'bool::from_str()'
// @has 'prim_associated_traits/struct.Number.html' '//a[@href="{{channel}}/std/primitive.str.html#method.eq"]' 'str::eq()'
// @has 'prim_associated_traits/struct.Number.html' '//a[@href="{{channel}}/std/primitive.never.html#method.eq"]' 'never::eq()'
/// [`f64::from_str()`] [`f32::from_str()`] [`isize::from_str()`] [`i8::from_str()`]
/// [`i16::from_str()`] [`i32::from_str()`] [`i64::from_str()`] [`i128::from_str()`]
/// [`u16::from_str()`] [`u32::from_str()`] [`u64::from_str()`] [`u128::from_str()`]
/// [`usize::from_str()`] [`u8::from_str()`] [`char::from_str()`] [`bool::from_str()`]
/// [`str::eq()`] [`never::eq()`]
pub struct Number {
    pub f_64: f64,
    pub f_32: f32,
    pub i_size: isize,
    pub i_8: i8,
    pub i_16: i16,
    pub i_32: i32,
    pub i_64: i64,
    pub i_128: i128,
    pub u_size: usize,
    pub u_8: u8,
    pub u_16: u16,
    pub u_32: u32,
    pub u_64: u64,
    pub u_128: u128,
    pub ch: char,
    pub boolean: bool,
    pub string: str,
    pub n: !,
}
