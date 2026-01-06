//! Regression test for <https://github.com/rust-lang/rust/issues/139596>
//!                     <https://github.com/rust-lang/rust/issues/136138>
//!                     <https://github.com/rust-lang/rust/issues/132985>

//@ check-pass

#![feature(
    min_generic_const_args,
    adt_const_params,
    generic_const_parameter_types,
    unsized_const_params
)]
#![expect(incomplete_features)]
use std::marker::{ConstParamTy, ConstParamTy_};
#[derive(ConstParamTy, PartialEq, Eq)]
struct Colour;

#[derive(ConstParamTy, PartialEq, Eq)]
enum A {
    B,
}

#[derive(ConstParamTy, PartialEq, Eq)]
enum MyOption<T> {
    #[allow(dead_code)]
    Some(T),
    None,
}

#[derive(ConstParamTy, PartialEq, Eq)]
struct Led<const C: Colour>;

#[derive(Eq, PartialEq, ConstParamTy)]
struct Foo<const N: usize>;

fn pass_enum<const P: MyOption<u32>>() {}

fn accepts_foo<const N: usize, const M: Foo<N>>() {}

fn accepts_bar<T: ConstParamTy_, const B: MyOption<T>>() {}

fn test<T: ConstParamTy_, const N: usize>() {
    accepts_foo::<N, { Foo::<N> }>();
    accepts_bar::<T, { MyOption::None::<T> }>();
}

fn main() {
    Led::<{ Colour }>;
}
