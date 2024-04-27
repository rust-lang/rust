//@ check-pass

#![feature(generic_const_items, associated_const_equality, adt_const_params)]
#![allow(incomplete_features)]

trait Owner {
    const C<const N: u32>: u32;
    const K<const N: u32>: u32;
    const Q<T>: Maybe<T>;
}

impl Owner for () {
    const C<const N: u32>: u32 = N;
    const K<const N: u32>: u32 = N + 1;
    const Q<T>: Maybe<T> = Maybe::Nothing;
}

fn take0<const N: u32>(_: impl Owner<C<N> = { N }>) {}
fn take1(_: impl Owner<K<99> = 100>) {}
fn take2(_: impl Owner<Q<()> = { Maybe::Just(()) }>) {}

fn main() {
    take0::<128>(());
    take1(());
}

#[derive(PartialEq, Eq, std::marker::ConstParamTy)]
enum Maybe<T> {
    Nothing,
    Just(T),
}
