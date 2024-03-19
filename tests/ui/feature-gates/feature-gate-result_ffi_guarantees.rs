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
    //~^ ERROR `extern` block uses type `Result
    fn result_fn_t(x: Result<extern "C" fn(), ()>);
    //~^ ERROR `extern` block uses type `Result
    fn result_nonnull_t(x: Result<std::ptr::NonNull<u8>, ()>);
    //~^ ERROR `extern` block uses type `Result
    fn result_unique_t(x: Result<std::ptr::Unique<u8>, ()>);
    //~^ ERROR `extern` block uses type `Result
    fn result_nonzero_u8_t(x: Result<num::NonZero<u8>, ()>);
    //~^ ERROR `extern` block uses type `Result
    fn result_nonzero_u16_t(x: Result<num::NonZero<u16>, ()>);
    //~^ ERROR `extern` block uses type `Result
    fn result_nonzero_u32_t(x: Result<num::NonZero<u32>, ()>);
    //~^ ERROR `extern` block uses type `Result
    fn result_nonzero_u64_t(x: Result<num::NonZero<u64>, ()>);
    //~^ ERROR `extern` block uses type `Result
    fn result_nonzero_usize_t(x: Result<num::NonZero<usize>, ()>);
    //~^ ERROR `extern` block uses type `Result
    fn result_nonzero_i8_t(x: Result<num::NonZero<i8>, ()>);
    //~^ ERROR `extern` block uses type `Result
    fn result_nonzero_i16_t(x: Result<num::NonZero<i16>, ()>);
    //~^ ERROR `extern` block uses type `Result
    fn result_nonzero_i32_t(x: Result<num::NonZero<i32>, ()>);
    //~^ ERROR `extern` block uses type `Result
    fn result_nonzero_i64_t(x: Result<num::NonZero<i64>, ()>);
    //~^ ERROR `extern` block uses type `Result
    fn result_nonzero_isize_t(x: Result<num::NonZero<isize>, ()>);
    //~^ ERROR `extern` block uses type `Result
    fn result_transparent_struct_t(x: Result<TransparentStruct<num::NonZero<u8>>, ()>);
    //~^ ERROR `extern` block uses type `Result
    fn result_transparent_enum_t(x: Result<TransparentEnum<num::NonZero<u8>>, ()>);
    //~^ ERROR `extern` block uses type `Result
    fn result_phantom_t(x: Result<num::NonZero<u8>, std::marker::PhantomData<()>>);
    //~^ ERROR `extern` block uses type `Result
    fn result_1zst_exhaustive_no_variant_t(x: Result<num::NonZero<u8>, Z>);
    //~^ ERROR `extern` block uses type `Result
    fn result_1zst_exhaustive_no_field_t(x: Result<num::NonZero<u8>, NoField>);
    //~^ ERROR `extern` block uses type `Result

    fn result_ref_e(x: Result<(), &'static u8>);
    //~^ ERROR `extern` block uses type `Result
    fn result_fn_e(x: Result<(), extern "C" fn()>);
    //~^ ERROR `extern` block uses type `Result
    fn result_nonnull_e(x: Result<(), std::ptr::NonNull<u8>>);
    //~^ ERROR `extern` block uses type `Result
    fn result_unique_e(x: Result<(), std::ptr::Unique<u8>>);
    //~^ ERROR `extern` block uses type `Result
    fn result_nonzero_u8_e(x: Result<(), num::NonZero<u8>>);
    //~^ ERROR `extern` block uses type `Result
    fn result_nonzero_u16_e(x: Result<(), num::NonZero<u16>>);
    //~^ ERROR `extern` block uses type `Result
    fn result_nonzero_u32_e(x: Result<(), num::NonZero<u32>>);
    //~^ ERROR `extern` block uses type `Result
    fn result_nonzero_u64_e(x: Result<(), num::NonZero<u64>>);
    //~^ ERROR `extern` block uses type `Result
    fn result_nonzero_usize_e(x: Result<(), num::NonZero<usize>>);
    //~^ ERROR `extern` block uses type `Result
    fn result_nonzero_i8_e(x: Result<(), num::NonZero<i8>>);
    //~^ ERROR `extern` block uses type `Result
    fn result_nonzero_i16_e(x: Result<(), num::NonZero<i16>>);
    //~^ ERROR `extern` block uses type `Result
    fn result_nonzero_i32_e(x: Result<(), num::NonZero<i32>>);
    //~^ ERROR `extern` block uses type `Result
    fn result_nonzero_i64_e(x: Result<(), num::NonZero<i64>>);
    //~^ ERROR `extern` block uses type `Result
    fn result_nonzero_isize_e(x: Result<(), num::NonZero<isize>>);
    //~^ ERROR `extern` block uses type `Result
    fn result_transparent_struct_e(x: Result<(), TransparentStruct<num::NonZero<u8>>>);
    //~^ ERROR `extern` block uses type `Result
    fn result_transparent_enum_e(x: Result<(), TransparentEnum<num::NonZero<u8>>>);
    //~^ ERROR `extern` block uses type `Result
    fn result_phantom_e(x: Result<num::NonZero<u8>, std::marker::PhantomData<()>>);
    //~^ ERROR `extern` block uses type `Result
    fn result_1zst_exhaustive_no_variant_e(x: Result<Z, num::NonZero<u8>>);
    //~^ ERROR `extern` block uses type `Result
    fn result_1zst_exhaustive_no_field_e(x: Result<NoField, num::NonZero<u8>>);
    //~^ ERROR `extern` block uses type `Result
}

pub fn main() {}
