//@ compile-flags: -Znext-solver=globally

// mismatched trait/impl const-generic param types
// (E0053) used to ICE in CTFE instead of stopping at the error (#161532)

#![feature(generic_const_args)]
#![feature(min_generic_const_args)]
#![feature(generic_const_items)]
#![allow(incomplete_features)]

trait Owner {
    const K<const N: u16>: u32;
}

impl Owner for () {
    const K<const N: u32>: u32 = N + 1;
    //~^ ERROR associated constant `K` has an incompatible generic parameter for trait `Owner`
}

fn main() {}
