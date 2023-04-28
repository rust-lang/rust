#![feature(transparent_unions)]
#![feature(ptr_internals)]
#![deny(improper_ctypes)]
#![allow(dead_code)]

use std::num;

enum Z {}
enum U {
    A,
}
enum B {
    C,
    D,
}
enum T {
    E,
    F,
    G,
}

#[repr(C)]
enum ReprC {
    A,
    B,
    C,
}

#[repr(u8)]
enum U8 {
    A,
    B,
    C,
}

#[repr(isize)]
enum Isize {
    A,
    B,
    C,
}

#[repr(transparent)]
struct TransparentStruct<T>(T, std::marker::PhantomData<Z>);

#[repr(transparent)]
enum TransparentEnum<T> {
    Variant(T, std::marker::PhantomData<Z>),
}

#[repr(transparent)]
union TransparentUnion<T: Copy> {
    field: T,
}

struct Rust<T>(T);

extern "C" {
    fn zf(x: Z);
    fn uf(x: U); //~ ERROR `extern` block uses type `U`
    fn bf(x: B); //~ ERROR `extern` block uses type `B`
    fn tf(x: T); //~ ERROR `extern` block uses type `T`
    fn repr_c(x: ReprC);
    fn repr_u8(x: U8);
    fn repr_isize(x: Isize);
    fn option_ref(x: Option<&'static u8>);
    fn option_fn(x: Option<extern "C" fn()>);
    fn nonnull(x: Option<std::ptr::NonNull<u8>>);
    fn unique(x: Option<std::ptr::Unique<u8>>);
    fn nonzero_u8(x: Option<num::NonZeroU8>);
    fn nonzero_u16(x: Option<num::NonZeroU16>);
    fn nonzero_u32(x: Option<num::NonZeroU32>);
    fn nonzero_u64(x: Option<num::NonZeroU64>);
    fn nonzero_u128(x: Option<num::NonZeroU128>);
    //~^ ERROR `extern` block uses type `u128`
    fn nonzero_usize(x: Option<num::NonZeroUsize>);
    fn nonzero_i8(x: Option<num::NonZeroI8>);
    fn nonzero_i16(x: Option<num::NonZeroI16>);
    fn nonzero_i32(x: Option<num::NonZeroI32>);
    fn nonzero_i64(x: Option<num::NonZeroI64>);
    fn nonzero_i128(x: Option<num::NonZeroI128>);
    //~^ ERROR `extern` block uses type `i128`
    fn result_nonzero_u8(x: Result<(), num::NonZeroU8>);
    fn result_nonzero_u16(x: Result<(), num::NonZeroU16>);
    fn result_nonzero_u32(x: Result<(), num::NonZeroU32>);
    fn result_nonzero_u64(x: Result<(), num::NonZeroU64>);
    fn result_nonzero_u128(x: Result<(), num::NonZeroU128>);
    //~^ ERROR `extern` block uses type `u128`
    fn result_nonzero_usize(x: Result<(), num::NonZeroUsize>);
    fn result_nonzero_i8(x: Result<(), num::NonZeroI8>);
    fn result_nonzero_i16(x: Result<(), num::NonZeroI16>);
    fn result_nonzero_i32(x: Result<(), num::NonZeroI32>);
    fn result_nonzero_i64(x: Result<(), num::NonZeroI64>);
    fn result_nonzero_i128(x: Result<(), num::NonZeroI128>);
    //~^ ERROR `extern` block uses type `i128`
    fn left_result_nonzero_u8(x: Result<num::NonZeroU8, ()>);
    fn left_result_nonzero_u16(x: Result<num::NonZeroU16, ()>);
    fn left_result_nonzero_u32(x: Result<num::NonZeroU32, ()>);
    fn left_result_nonzero_u64(x: Result<num::NonZeroU64, ()>);
    fn left_result_nonzero_u128(x: Result<num::NonZeroU128, ()>);
    //~^ ERROR `extern` block uses type `u128`
    fn left_result_nonzero_usize(x: Result<num::NonZeroUsize, ()>);
    fn left_result_nonzero_i8(x: Result<num::NonZeroI8, ()>);
    fn left_result_nonzero_i16(x: Result<num::NonZeroI16, ()>);
    fn left_result_nonzero_i32(x: Result<num::NonZeroI32, ()>);
    fn left_result_nonzero_i64(x: Result<num::NonZeroI64, ()>);
    fn left_result_nonzero_i128(x: Result<num::NonZeroI128, ()>);
    //~^ ERROR `extern` block uses type `i128`
    fn result_sized(x: Result<i32, num::NonZeroI128>);
    //~^ ERROR `extern` block uses type `Result<i32, NonZeroI128>`
    fn left_result_sized(x: Result<num::NonZeroI128, i32>);
    //~^ ERROR `extern` block uses type `Result<NonZeroI128, i32>`
    fn nonzero_isize(x: Option<num::NonZeroIsize>);
    fn transparent_struct(x: Option<TransparentStruct<num::NonZeroU8>>);
    fn transparent_enum(x: Option<TransparentEnum<num::NonZeroU8>>);
    fn transparent_union(x: Option<TransparentUnion<num::NonZeroU8>>);
    //~^ ERROR `extern` block uses type
    fn repr_rust(x: Option<Rust<num::NonZeroU8>>); //~ ERROR `extern` block uses type
}

pub fn main() {}
