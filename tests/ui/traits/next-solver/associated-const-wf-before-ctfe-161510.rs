//@ compile-flags: -Znext-solver=globally

#![feature(generic_const_items)]
#![feature(min_generic_const_args)]
#![expect(incomplete_features)]

trait Owner {
    type const K<const N: u16>: u16;
}

impl Owner for () {
    type const K<const N: u32>: u32 = N;
    //~^ ERROR associated constant `K` has an incompatible generic parameter for trait `Owner`
}

fn take1(_: impl Owner<K<9> = 0>) {}

fn main() {
    take1(());
    //~^ ERROR type mismatch resolving `<() as Owner>::K<9> == 0`
}
