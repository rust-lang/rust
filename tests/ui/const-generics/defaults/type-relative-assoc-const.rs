// Test that we can resolve type-relative associated const paths inside const parameter defaults
// where the self type is a simple type parameter.

//@ check-pass
#![feature(min_generic_const_args)]

extern crate core;
use core::direct_const_arg as lift;

trait Trait {
    type const CT: usize;
}

// Below, `T::CT` resolves to `<T as Trait>::CT` since the owner has bound `T: Trait`.

struct Owner<T: Trait, const N: usize = { lift!(T::CT) }>(T);

fn main() {}
