//@ run-pass
//! Test using `#[splat]` on tuple arguments with where clause bounds.

#![allow(incomplete_features)]
#![feature(splat)]
#![feature(tuple_trait)]

fn where_splat<T>(#[splat] _t: T) where T: std::marker::Tuple {}

fn where_splat_with_extra<T>(#[splat] _t: T, _extra: u32) where T: std::marker::Tuple {}

fn impl_tuple_splat(#[splat] _t: impl std::marker::Tuple) {}

fn impl_tuple_splat_with_extra(#[splat] _t: impl std::marker::Tuple, _extra: u32) {}

fn main() {
    // empty tuple
    where_splat();

    // single element
    where_splat(1u32);
    where_splat(1);

    // two elements
    where_splat(1u32, 2i8);
    where_splat(1, 2);

    // three elements
    where_splat(1u32, 2i8, 3.0f64);
    where_splat(1, 2, 3.0);

    // with extra non-splatted arg
    where_splat_with_extra(1u32, 2i8, 42u32);
    where_splat_with_extra(1, 2, 42);

    // impl Trait syntax variants
    impl_tuple_splat();
    impl_tuple_splat(1u32);
    impl_tuple_splat(1u32, 2i8);
    impl_tuple_splat(1, 2, 3.0);

    impl_tuple_splat_with_extra(1u32, 2i8, 42u32);
    impl_tuple_splat_with_extra(1, 2, 42);
}
