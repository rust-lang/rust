// Regression test for issue #108271.
// Detect and reject generic params in the type of assoc consts used in an equality bound.
#![feature(associated_const_equality)]

trait Trait<'a, T, const N: usize> {
    const K: &'a [T; N];
}

fn take0<'r>(_: impl Trait<'r, (), 0, K = { &[] }>) {}
//~^ ERROR the type of the associated constant `K` must not depend on generic parameters
//~| NOTE its type must not depend on the lifetime parameter `'r`
//~| NOTE the lifetime parameter `'r` is defined here
fn take1<A>(_: impl Trait<'static, A, 0, K = { &[] }>) {}
//~^ ERROR the type of the associated constant `K` must not depend on generic parameters
//~| NOTE its type must not depend on the type parameter `A`
//~| NOTE the type parameter `A` is defined here
fn take2<const Q: usize>(_: impl Trait<'static, (), Q, K = { [] }>) {}
//~^ ERROR the type of the associated constant `K` must not depend on generic parameters
//~| NOTE its type must not depend on the const parameter `Q`
//~| NOTE the const parameter `Q` is defined here

trait Project {
    const S: Self;
}

// FIXME(associated_const_equality): The error messages below aren't super great at the moment:
// Here, `Self` is a type parameter of the trait `Project`, not of the function `take3`
// unlike the cases above. We should mention the APIT / the parameter `P` instead.

fn take3(_: impl Project<S = {}>) {}
//~^ ERROR the type of the associated constant `S` must not depend on generic parameters
//~| NOTE its type must not depend on the type parameter `Self`

fn take4<P: Project<S = {}>>(_: P) {}
//~^ ERROR the type of the associated constant `S` must not depend on generic parameters
//~| NOTE its type must not depend on the type parameter `Self`

fn main() {}
