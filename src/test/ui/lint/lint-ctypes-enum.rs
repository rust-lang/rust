#![feature(transparent_enums, transparent_unions)]
#![deny(improper_ctypes)]
#![allow(dead_code)]

use std::num;

enum Z { }
enum U { A }
enum B { C, D }
enum T { E, F, G }

#[repr(C)]
enum ReprC { A, B, C }

#[repr(u8)]
enum U8 { A, B, C }

#[repr(isize)]
enum Isize { A, B, C }

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

extern {
   fn zf(x: Z);
   fn uf(x: U); //~ ERROR enum has no representation hint
   fn bf(x: B); //~ ERROR enum has no representation hint
   fn tf(x: T); //~ ERROR enum has no representation hint
   fn repr_c(x: ReprC);
   fn repr_u8(x: U8);
   fn repr_isize(x: Isize);
   fn option_ref(x: Option<&'static u8>);
   fn option_fn(x: Option<extern "C" fn()>);
   fn nonnull(x: Option<std::ptr::NonNull<u8>>);
   fn nonzero_u8(x: Option<num::NonZeroU8>);
   fn nonzero_u16(x: Option<num::NonZeroU16>);
   fn nonzero_u32(x: Option<num::NonZeroU32>);
   fn nonzero_u64(x: Option<num::NonZeroU64>);
   fn nonzero_u128(x: Option<num::NonZeroU128>);
   //~^ ERROR 128-bit integers don't currently have a known stable ABI
   fn nonzero_usize(x: Option<num::NonZeroUsize>);
   fn nonzero_i8(x: Option<num::NonZeroI8>);
   fn nonzero_i16(x: Option<num::NonZeroI16>);
   fn nonzero_i32(x: Option<num::NonZeroI32>);
   fn nonzero_i64(x: Option<num::NonZeroI64>);
   fn nonzero_i128(x: Option<num::NonZeroI128>);
   //~^ ERROR 128-bit integers don't currently have a known stable ABI
   fn nonzero_isize(x: Option<num::NonZeroIsize>);
   fn transparent_struct(x: Option<TransparentStruct<num::NonZeroU8>>);
   fn transparent_enum(x: Option<TransparentEnum<num::NonZeroU8>>);
   fn transparent_union(x: Option<TransparentUnion<num::NonZeroU8>>);
   //~^ ERROR enum has no representation hint
   fn repr_rust(x: Option<Rust<num::NonZeroU8>>); //~ ERROR enum has no representation hint
   fn no_result(x: Result<(), num::NonZeroI32>); //~ ERROR enum has no representation hint
}

pub fn main() { }
