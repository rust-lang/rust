#![allow(dead_code)]
#![deny(improper_ctypes)]
#![feature(generic_nonzero)]
#![feature(ptr_internals)]
#![feature(transparent_unions)]

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
   fn nonzero_u8(x: Option<num::NonZero<u8>>);
   fn nonzero_u16(x: Option<num::NonZero<u16>>);
   fn nonzero_u32(x: Option<num::NonZero<u32>>);
   fn nonzero_u64(x: Option<num::NonZero<u64>>);
   fn nonzero_u128(x: Option<num::NonZero<u128>>);
   //~^ ERROR `extern` block uses type `u128`
   fn nonzero_usize(x: Option<num::NonZero<usize>>);
   fn nonzero_i8(x: Option<num::NonZero<i8>>);
   fn nonzero_i16(x: Option<num::NonZero<i16>>);
   fn nonzero_i32(x: Option<num::NonZero<i32>>);
   fn nonzero_i64(x: Option<num::NonZero<i64>>);
   fn nonzero_i128(x: Option<num::NonZero<i128>>);
   //~^ ERROR `extern` block uses type `i128`
   fn nonzero_isize(x: Option<num::NonZero<isize>>);
   fn transparent_struct(x: Option<TransparentStruct<num::NonZero<u8>>>);
   fn transparent_enum(x: Option<TransparentEnum<num::NonZero<u8>>>);
   fn transparent_union(x: Option<TransparentUnion<num::NonZero<u8>>>);
   //~^ ERROR `extern` block uses type
   fn repr_rust(x: Option<Rust<num::NonZero<u8>>>); //~ ERROR `extern` block uses type
   fn no_result(x: Result<(), num::NonZero<i32>>); //~ ERROR `extern` block uses type
}

pub fn main() {}
