//@ compile-flags: -Znext-solver

#![feature(min_generic_const_args, generic_const_args)]
#![expect(incomplete_features)]

use std::gca;

trait Trait {
    const PROJECTED_A: usize;
    const PROJECTED_B: usize;
}

struct StructImpl;
struct GenericStructImpl<const N: usize>;

impl Trait for StructImpl {
    const PROJECTED_A: usize = 1;
    const PROJECTED_B: usize = 1;
}

impl<const N: usize> Trait for GenericStructImpl<N> {
    const PROJECTED_A: usize = N;
    const PROJECTED_B: usize = N;
}

const FREE_A: usize = 1;
const FREE_B: usize = 1;

struct Struct<const N: usize>;

fn f<const N: usize>() {
    let _: Struct<{ gca!(<GenericStructImpl<N> as Trait>::PROJECTED_A) }> =
        Struct::<{ gca!(<GenericStructImpl<N> as Trait>::PROJECTED_B) }>;
    //~^ ERROR mismatched types
}

fn g<T: Trait>() {
    let _: Struct<{ gca!(T::PROJECTED_A) }> = Struct::<{ gca!(T::PROJECTED_B) }>;
    //~^ ERROR mismatched types
}

fn main() {
    f::<2>();
    g::<StructImpl>();
}
