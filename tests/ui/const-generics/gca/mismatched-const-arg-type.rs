//! Regression test for <https://github.com/rust-lang/rust/issues/156780>.

//@ compile-flags: -Znext-solver

#![feature(
    generic_const_args,
    generic_const_items,
    macroless_generic_const_args,
    min_generic_const_args
)]
#![expect(incomplete_features)]

const F: f64 = 1.0;

const ADD1<const N: usize>: usize = N + 1;

const IGNORE<const N: usize>: usize = 0;

trait Tr {
    const C<const N: usize>: usize;
}

impl Tr for () {
    const C<const N: usize>: usize = N + 1;
}

fn main() {
    let _: [(); ADD1::<1f64>];
    //~^ ERROR: the constant `1f64` is not of type `usize`
    //~| ERROR: is not well-formed
    let _: [(); ADD1::<1u8>];
    //~^ ERROR: the constant `1` is not of type `usize`
    //~| ERROR: is not well-formed
    let _: [(); ADD1::<1i32>];
    //~^ ERROR: the constant `1` is not of type `usize`
    //~| ERROR: is not well-formed
    let _: [(); ADD1::<true>];
    //~^ ERROR: the constant `true` is not of type `usize`
    //~| ERROR: is not well-formed
    let _: [(); ADD1::<'a'>];
    //~^ ERROR: the constant `'a'` is not of type `usize`
    //~| ERROR: is not well-formed
    let _: [(); ADD1::<"s">];
    //~^ ERROR: the constant `"s"` is not of type `usize`
    //~| ERROR: is not well-formed
    let _: [(); ADD1::<F>];
    //~^ ERROR: the constant `1f64` is not of type `usize`
    //~| ERROR: is not well-formed
    let _: [(); IGNORE::<1f64>];
    //~^ ERROR: the constant `1f64` is not of type `usize`
    //~| ERROR: is not well-formed
    let _: [(); <() as Tr>::C::<1f64>];
    //~^ ERROR: type mismatch resolving `<() as Tr>::C<1f64> == _`
    //~| ERROR: is not well-formed
}
