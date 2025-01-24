//@ known-bug: #118778
//@ edition:2021
//@ needs-rustc-debug-assertions

#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

trait Owner {
    type T<const N: u16>;
}

impl Owner for () {
    type T<const N: u32> = U32<{ N + 1 }>
    where
        U32<{ N + 1 }>:;
}

struct U32<const N: u32>;

fn take1(_: impl Owner<T<1> = U32<1>>) {}

fn main() {
    take1(());
}
