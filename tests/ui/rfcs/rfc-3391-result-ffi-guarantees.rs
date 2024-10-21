//@ build-pass
/*!

The guarantees in RFC 3391 were strengthened as a result of the 2024 Oct 09 T-lang meeting[^1]
following the precedent of T-lang's guaranteeing[^2] ABI compatibility for "Option-like" enums[^2].
We now guarantee ABI compatibility for enums that conform to these rules described by scottmcm:

* The enum `E` has exactly two variants.
* One variant has exactly one field, of type `T`.
* `T` is a `rustc_nonnull_optimization_guaranteed` type.
* All fields of the other variant are 1-ZSTs.

Where "all" fields includes "there aren't any fields, so they're vacuously all 1-ZSTs".

Note: "1-ZST" means a type of size 0 and alignment 1.

The reason alignment of the zero-sized type matters is it can affect the alignment of the enum,
which also will affect its size if the enum has a non-zero size.

[^1]: <https://github.com/rust-lang/rust/pull/130628#issuecomment-2402761599>
[^2]: <https://github.com/rust-lang/rust/pull/60300#issuecomment-487000474>

*/

#![allow(dead_code)]
#![deny(improper_ctypes)]
#![feature(ptr_internals)]

use std::num;

enum Z {}

#[repr(transparent)]
struct TransparentStruct<T>(T, std::marker::PhantomData<Z>);

#[repr(transparent)]
enum TransparentEnum<T> {
    Variant(T, std::marker::PhantomData<Z>),
}

struct NoField;

extern "C" {
    fn result_ref_t(x: Result<&'static u8, ()>);
    fn result_fn_t(x: Result<extern "C" fn(), ()>);
    fn result_nonnull_t(x: Result<std::ptr::NonNull<u8>, ()>);
    fn result_unique_t(x: Result<std::ptr::Unique<u8>, ()>);
    fn result_nonzero_u8_t(x: Result<num::NonZero<u8>, ()>);
    fn result_nonzero_u16_t(x: Result<num::NonZero<u16>, ()>);
    fn result_nonzero_u32_t(x: Result<num::NonZero<u32>, ()>);
    fn result_nonzero_u64_t(x: Result<num::NonZero<u64>, ()>);
    fn result_nonzero_usize_t(x: Result<num::NonZero<usize>, ()>);
    fn result_nonzero_i8_t(x: Result<num::NonZero<i8>, ()>);
    fn result_nonzero_i16_t(x: Result<num::NonZero<i16>, ()>);
    fn result_nonzero_i32_t(x: Result<num::NonZero<i32>, ()>);
    fn result_nonzero_i64_t(x: Result<num::NonZero<i64>, ()>);
    fn result_nonzero_isize_t(x: Result<num::NonZero<isize>, ()>);
    fn result_transparent_struct_t(x: Result<TransparentStruct<num::NonZero<u8>>, ()>);
    fn result_transparent_enum_t(x: Result<TransparentEnum<num::NonZero<u8>>, ()>);
    fn result_phantom_t(x: Result<num::NonZero<u8>, std::marker::PhantomData<()>>);
    fn result_1zst_exhaustive_no_variant_t(x: Result<num::NonZero<u8>, Z>);
    fn result_1zst_exhaustive_no_field_t(x: Result<num::NonZero<u8>, NoField>);

    fn result_ref_e(x: Result<(), &'static u8>);
    fn result_fn_e(x: Result<(), extern "C" fn()>);
    fn result_nonnull_e(x: Result<(), std::ptr::NonNull<u8>>);
    fn result_unique_e(x: Result<(), std::ptr::Unique<u8>>);
    fn result_nonzero_u8_e(x: Result<(), num::NonZero<u8>>);
    fn result_nonzero_u16_e(x: Result<(), num::NonZero<u16>>);
    fn result_nonzero_u32_e(x: Result<(), num::NonZero<u32>>);
    fn result_nonzero_u64_e(x: Result<(), num::NonZero<u64>>);
    fn result_nonzero_usize_e(x: Result<(), num::NonZero<usize>>);
    fn result_nonzero_i8_e(x: Result<(), num::NonZero<i8>>);
    fn result_nonzero_i16_e(x: Result<(), num::NonZero<i16>>);
    fn result_nonzero_i32_e(x: Result<(), num::NonZero<i32>>);
    fn result_nonzero_i64_e(x: Result<(), num::NonZero<i64>>);
    fn result_nonzero_isize_e(x: Result<(), num::NonZero<isize>>);
    fn result_transparent_struct_e(x: Result<(), TransparentStruct<num::NonZero<u8>>>);
    fn result_transparent_enum_e(x: Result<(), TransparentEnum<num::NonZero<u8>>>);
    fn result_phantom_e(x: Result<num::NonZero<u8>, std::marker::PhantomData<()>>);
    fn result_1zst_exhaustive_no_variant_e(x: Result<Z, num::NonZero<u8>>);
    fn result_1zst_exhaustive_no_field_e(x: Result<NoField, num::NonZero<u8>>);
}

// Custom "Result-like" enum for testing custom "Option-like" types are also accepted
enum Either<L, R> {
    Left(L),
    Right(R),
}

extern "C" {
    fn either_ref_t(x: Either<&'static u8, ()>);
    fn either_fn_t(x: Either<extern "C" fn(), ()>);
    fn either_nonnull_t(x: Either<std::ptr::NonNull<u8>, ()>);
    fn either_unique_t(x: Either<std::ptr::Unique<u8>, ()>);
    fn either_nonzero_u8_t(x: Either<num::NonZero<u8>, ()>);
    fn either_nonzero_u16_t(x: Either<num::NonZero<u16>, ()>);
    fn either_nonzero_u32_t(x: Either<num::NonZero<u32>, ()>);
    fn either_nonzero_u64_t(x: Either<num::NonZero<u64>, ()>);
    fn either_nonzero_usize_t(x: Either<num::NonZero<usize>, ()>);
    fn either_nonzero_i8_t(x: Either<num::NonZero<i8>, ()>);
    fn either_nonzero_i16_t(x: Either<num::NonZero<i16>, ()>);
    fn either_nonzero_i32_t(x: Either<num::NonZero<i32>, ()>);
    fn either_nonzero_i64_t(x: Either<num::NonZero<i64>, ()>);
    fn either_nonzero_isize_t(x: Either<num::NonZero<isize>, ()>);
    fn either_transparent_struct_t(x: Either<TransparentStruct<num::NonZero<u8>>, ()>);
    fn either_transparent_enum_t(x: Either<TransparentEnum<num::NonZero<u8>>, ()>);
    fn either_phantom_t(x: Either<num::NonZero<u8>, std::marker::PhantomData<()>>);
    fn either_1zst_exhaustive_no_variant_t(x: Either<num::NonZero<u8>, Z>);
    fn either_1zst_exhaustive_no_field_t(x: Either<num::NonZero<u8>, NoField>);

    fn either_ref_e(x: Either<(), &'static u8>);
    fn either_fn_e(x: Either<(), extern "C" fn()>);
    fn either_nonnull_e(x: Either<(), std::ptr::NonNull<u8>>);
    fn either_unique_e(x: Either<(), std::ptr::Unique<u8>>);
    fn either_nonzero_u8_e(x: Either<(), num::NonZero<u8>>);
    fn either_nonzero_u16_e(x: Either<(), num::NonZero<u16>>);
    fn either_nonzero_u32_e(x: Either<(), num::NonZero<u32>>);
    fn either_nonzero_u64_e(x: Either<(), num::NonZero<u64>>);
    fn either_nonzero_usize_e(x: Either<(), num::NonZero<usize>>);
    fn either_nonzero_i8_e(x: Either<(), num::NonZero<i8>>);
    fn either_nonzero_i16_e(x: Either<(), num::NonZero<i16>>);
    fn either_nonzero_i32_e(x: Either<(), num::NonZero<i32>>);
    fn either_nonzero_i64_e(x: Either<(), num::NonZero<i64>>);
    fn either_nonzero_isize_e(x: Either<(), num::NonZero<isize>>);
    fn either_transparent_struct_e(x: Either<(), TransparentStruct<num::NonZero<u8>>>);
    fn either_transparent_enum_e(x: Either<(), TransparentEnum<num::NonZero<u8>>>);
    fn either_phantom_e(x: Either<num::NonZero<u8>, std::marker::PhantomData<()>>);
    fn either_1zst_exhaustive_no_variant_e(x: Either<Z, num::NonZero<u8>>);
    fn either_1zst_exhaustive_no_field_e(x: Either<NoField, num::NonZero<u8>>);
}

pub fn main() {}
