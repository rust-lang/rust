// check-pass

#![feature(generic_const_items, associated_const_equality)]
#![allow(incomplete_features)]

trait Owner {
    const C<const N: u32>: u32;
    const K<const N: u32>: u32;
}

impl Owner for () {
    const C<const N: u32>: u32 = N;
    const K<const N: u32>: u32 = N + 1;
}

fn take0<const N: u32>(_: impl Owner<C<N> = { N }>) {}
fn take1(_: impl Owner<K<99> = 100>) {}

fn main() {
    take0::<128>(());
    take1(());
}
