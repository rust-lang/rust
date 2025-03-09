#![allow(dead_code)]
#![deny(improper_ctypes)]
#![feature(ptr_internals)]
#![feature(transparent_unions)]
#![feature(repr128)]
#![allow(incomplete_features)]

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

#[repr(u128)]
enum U128 {
    A,
    B,
    C,
}

#[repr(i128)]
enum I128 {
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

struct NoField;

#[repr(transparent)]
struct Field(());

#[non_exhaustive]
enum NonExhaustive {}

extern "C" {
    fn zf(x: Z);
    fn uf(x: U); //~ ERROR `extern` block uses type `U`
    fn bf(x: B); //~ ERROR `extern` block uses type `B`
    fn tf(x: T); //~ ERROR `extern` block uses type `T`
    fn repr_c(x: ReprC);
    fn repr_u8(x: U8);
    fn repr_isize(x: Isize);
    fn repr_u128(x: U128); //~ ERROR `extern` block uses type `U128`
    fn repr_i128(x: I128); //~ ERROR `extern` block uses type `I128`
    fn option_ref(x: Option<&'static u8>);
    fn option_fn(x: Option<extern "C" fn()>);
    fn option_nonnull(x: Option<std::ptr::NonNull<u8>>);
    fn option_unique(x: Option<std::ptr::Unique<u8>>);
    fn option_nonzero_u8(x: Option<num::NonZero<u8>>);
    fn option_nonzero_u16(x: Option<num::NonZero<u16>>);
    fn option_nonzero_u32(x: Option<num::NonZero<u32>>);
    fn option_nonzero_u64(x: Option<num::NonZero<u64>>);
    fn option_nonzero_u128(x: Option<num::NonZero<u128>>);
    //~^ ERROR `extern` block uses type `u128`
    fn option_nonzero_usize(x: Option<num::NonZero<usize>>);
    fn option_nonzero_i8(x: Option<num::NonZero<i8>>);
    fn option_nonzero_i16(x: Option<num::NonZero<i16>>);
    fn option_nonzero_i32(x: Option<num::NonZero<i32>>);
    fn option_nonzero_i64(x: Option<num::NonZero<i64>>);
    fn option_nonzero_i128(x: Option<num::NonZero<i128>>);
    //~^ ERROR `extern` block uses type `i128`
    fn option_nonzero_isize(x: Option<num::NonZero<isize>>);
    fn option_transparent_struct(x: Option<TransparentStruct<num::NonZero<u8>>>);
    fn option_transparent_enum(x: Option<TransparentEnum<num::NonZero<u8>>>);
    fn option_transparent_union(x: Option<TransparentUnion<num::NonZero<u8>>>);
    //~^ ERROR `extern` block uses type
    fn option_repr_rust(x: Option<Rust<num::NonZero<u8>>>); //~ ERROR `extern` block uses type
    fn option_u8(x: Option<u8>); //~ ERROR `extern` block uses type

    fn result_ref_t(x: Result<&'static u8, ()>);
    fn result_fn_t(x: Result<extern "C" fn(), ()>);
    fn result_nonnull_t(x: Result<std::ptr::NonNull<u8>, ()>);
    fn result_unique_t(x: Result<std::ptr::Unique<u8>, ()>);
    fn result_nonzero_u8_t(x: Result<num::NonZero<u8>, ()>);
    fn result_nonzero_u16_t(x: Result<num::NonZero<u16>, ()>);
    fn result_nonzero_u32_t(x: Result<num::NonZero<u32>, ()>);
    fn result_nonzero_u64_t(x: Result<num::NonZero<u64>, ()>);
    fn result_nonzero_u128_t(x: Result<num::NonZero<u128>, ()>);
    //~^ ERROR `extern` block uses type `u128`
    fn result_nonzero_usize_t(x: Result<num::NonZero<usize>, ()>);
    fn result_nonzero_i8_t(x: Result<num::NonZero<i8>, ()>);
    fn result_nonzero_i16_t(x: Result<num::NonZero<i16>, ()>);
    fn result_nonzero_i32_t(x: Result<num::NonZero<i32>, ()>);
    fn result_nonzero_i64_t(x: Result<num::NonZero<i64>, ()>);
    fn result_nonzero_i128_t(x: Result<num::NonZero<i128>, ()>);
    //~^ ERROR `extern` block uses type `i128`
    fn result_nonzero_isize_t(x: Result<num::NonZero<isize>, ()>);
    fn result_transparent_struct_t(x: Result<TransparentStruct<num::NonZero<u8>>, ()>);
    fn result_transparent_enum_t(x: Result<TransparentEnum<num::NonZero<u8>>, ()>);
    fn result_transparent_union_t(x: Result<TransparentUnion<num::NonZero<u8>>, ()>);
    //~^ ERROR `extern` block uses type
    fn result_repr_rust_t(x: Result<Rust<num::NonZero<u8>>, ()>);
    //~^ ERROR `extern` block uses type
    fn result_phantom_t(x: Result<num::NonZero<u8>, std::marker::PhantomData<()>>);
    fn result_1zst_exhaustive_no_variant_t(x: Result<num::NonZero<u8>, Z>);
    fn result_1zst_exhaustive_single_variant_t(x: Result<num::NonZero<u8>, U>);
    //~^ ERROR `extern` block uses type
    fn result_1zst_exhaustive_multiple_variant_t(x: Result<num::NonZero<u8>, B>);
    //~^ ERROR `extern` block uses type
    fn result_1zst_non_exhaustive_no_variant_t(x: Result<num::NonZero<u8>, NonExhaustive>);
    //~^ ERROR `extern` block uses type
    fn result_1zst_exhaustive_no_field_t(x: Result<num::NonZero<u8>, NoField>);
    fn result_1zst_exhaustive_single_field_t(x: Result<num::NonZero<u8>, Field>);
    //~^ ERROR `extern` block uses type
    fn result_cascading_t(x: Result<Result<(), num::NonZero<u8>>, ()>);
    //~^ ERROR `extern` block uses type

    fn result_ref_e(x: Result<(), &'static u8>);
    fn result_fn_e(x: Result<(), extern "C" fn()>);
    fn result_nonnull_e(x: Result<(), std::ptr::NonNull<u8>>);
    fn result_unique_e(x: Result<(), std::ptr::Unique<u8>>);
    fn result_nonzero_u8_e(x: Result<(), num::NonZero<u8>>);
    fn result_nonzero_u16_e(x: Result<(), num::NonZero<u16>>);
    fn result_nonzero_u32_e(x: Result<(), num::NonZero<u32>>);
    fn result_nonzero_u64_e(x: Result<(), num::NonZero<u64>>);
    fn result_nonzero_u128_e(x: Result<(), num::NonZero<u128>>);
    //~^ ERROR `extern` block uses type `u128`
    fn result_nonzero_usize_e(x: Result<(), num::NonZero<usize>>);
    fn result_nonzero_i8_e(x: Result<(), num::NonZero<i8>>);
    fn result_nonzero_i16_e(x: Result<(), num::NonZero<i16>>);
    fn result_nonzero_i32_e(x: Result<(), num::NonZero<i32>>);
    fn result_nonzero_i64_e(x: Result<(), num::NonZero<i64>>);
    fn result_nonzero_i128_e(x: Result<(), num::NonZero<i128>>);
    //~^ ERROR `extern` block uses type `i128`
    fn result_nonzero_isize_e(x: Result<(), num::NonZero<isize>>);
    fn result_transparent_struct_e(x: Result<(), TransparentStruct<num::NonZero<u8>>>);
    fn result_transparent_enum_e(x: Result<(), TransparentEnum<num::NonZero<u8>>>);
    fn result_transparent_union_e(x: Result<(), TransparentUnion<num::NonZero<u8>>>);
    //~^ ERROR `extern` block uses type
    fn result_repr_rust_e(x: Result<(), Rust<num::NonZero<u8>>>);
    //~^ ERROR `extern` block uses type
    fn result_phantom_e(x: Result<num::NonZero<u8>, std::marker::PhantomData<()>>);
    fn result_1zst_exhaustive_no_variant_e(x: Result<Z, num::NonZero<u8>>);
    fn result_1zst_exhaustive_single_variant_e(x: Result<U, num::NonZero<u8>>);
    //~^ ERROR `extern` block uses type
    fn result_1zst_exhaustive_multiple_variant_e(x: Result<B, num::NonZero<u8>>);
    //~^ ERROR `extern` block uses type
    fn result_1zst_non_exhaustive_no_variant_e(x: Result<NonExhaustive, num::NonZero<u8>>);
    //~^ ERROR `extern` block uses type
    fn result_1zst_exhaustive_no_field_e(x: Result<NoField, num::NonZero<u8>>);
    fn result_1zst_exhaustive_single_field_e(x: Result<Field, num::NonZero<u8>>);
    //~^ ERROR `extern` block uses type
    fn result_cascading_e(x: Result<(), Result<(), num::NonZero<u8>>>);
    //~^ ERROR `extern` block uses type
    fn result_unit_t_e(x: Result<(), ()>);
    //~^ ERROR `extern` block uses type
}

pub fn main() {}
