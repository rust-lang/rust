//@ check-pass

#![feature(generic_const_items, min_generic_const_args)]
#![feature(adt_const_params, unsized_const_params, generic_const_parameter_types)]
#![expect(incomplete_features)]

use std::marker::{ConstParamTy, ConstParamTy_};

trait Owner {
    type const C<const N: u32>: u32;
    type const K<const N: u32>: u32;
    type const Q<T: ConstParamTy_>: Maybe<T>;
}

impl Owner for () {
    type const C<const N: u32>: u32 = N;
    type const K<const N: u32>: u32 = const { 99 + 1 };
    type const Q<T: ConstParamTy_>: Maybe<T> = Maybe::Nothing::<T>;
}

fn take0<const N: u32>(_: impl Owner<C<N> = { N }>) {}
fn take1(_: impl Owner<K<99> = 100>) {}
fn take2(_: impl Owner<Q<()> = { Maybe::Just::<()>(()) }>) {}

fn main() {
    take0::<128>(());
    take1(());
}

#[derive(PartialEq, Eq, ConstParamTy)]
enum Maybe<T> {
    Nothing,
    Just(T),
}
