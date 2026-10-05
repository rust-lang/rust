//@ check-pass

#![feature(generic_const_items, gca_min_const_items)]
#![feature(adt_const_params, const_param_ty_trait, generic_const_parameter_types)]
#![expect(incomplete_features)]

use std::gca;
use std::marker::{ConstParamTy, ConstParamTy_};

trait Owner {
    #[rustc_always_gca]
    const C<const N: u32>: u32;
    #[rustc_always_gca]
    const K<const N: u32>: u32;
    #[rustc_always_gca]
    const Q<T: ConstParamTy_>: Maybe<T>;
}

impl Owner for () {
    const C<const N: u32>: u32 = gca!(N);
    const K<const N: u32>: u32 = gca!(const { 99 + 1 });
    const Q<T: ConstParamTy_>: Maybe<T> = gca!(Maybe::Nothing::<T>);
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
