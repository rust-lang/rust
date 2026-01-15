// Regression test for issue #108271.
// Detect and reject generic params in the type of assoc consts used in an equality bound.
#![feature(
    min_generic_const_args,
    adt_const_params,
    unsized_const_params,
    generic_const_parameter_types,
)]
#![allow(incomplete_features)]

use std::marker::ConstParamTy_;

trait Trait<'a, T: 'a + ConstParamTy_, const N: usize> {
    #[type_const]
    const K: &'a [T; N];
}

fn take0<'r, A: 'r + ConstParamTy_, const Q: usize>(
    //~^ NOTE the lifetime parameter `'r` is defined here
    //~| NOTE the type parameter `A` is defined here
    //~| NOTE the const parameter `Q` is defined here
    _: impl Trait<'r, A, Q, K = const { loop {} }>
    //~^ ERROR the type of the associated constant `K` must not depend on generic parameters
    //~| NOTE its type must not depend on the lifetime parameter `'r`
    //~| NOTE `K` has type `&'r [A; Q]`
    //~| ERROR the type of the associated constant `K` must not depend on generic parameters
    //~| NOTE its type must not depend on the type parameter `A`
    //~| NOTE `K` has type `&'r [A; Q]`
    //~| ERROR the type of the associated constant `K` must not depend on generic parameters
    //~| NOTE its type must not depend on the const parameter `Q`
    //~| NOTE `K` has type `&'r [A; Q]`
) {}

trait Project: ConstParamTy_ {
    #[type_const]
    const SELF: Self;
}

fn take1(_: impl Project<SELF = const {}>) {}
//~^ ERROR the type of the associated constant `SELF` must not depend on `impl Trait`
//~| NOTE its type must not depend on `impl Trait`
//~| NOTE the `impl Trait` is specified here

fn take2<P: Project<SELF = const {}>>(_: P) {}
//~^ ERROR the type of the associated constant `SELF` must not depend on generic parameters
//~| NOTE its type must not depend on the type parameter `P`
//~| NOTE the type parameter `P` is defined here
//~| NOTE `SELF` has type `P`

trait Iface<'r>: ConstParamTy_ {
    //~^ NOTE the lifetime parameter `'r` is defined here
    //~| NOTE the lifetime parameter `'r` is defined here
    type Assoc<const Q: usize>: Trait<'r, Self, Q, K = const { loop {} }>
    //~^ ERROR the type of the associated constant `K` must not depend on generic parameters
    //~| ERROR the type of the associated constant `K` must not depend on generic parameters
    //~| NOTE its type must not depend on the lifetime parameter `'r`
    //~| NOTE `K` has type `&'r [Self; Q]`
    //~| ERROR the type of the associated constant `K` must not depend on `Self`
    //~| NOTE its type must not depend on `Self`
    //~| NOTE `K` has type `&'r [Self; Q]`
    //~| ERROR the type of the associated constant `K` must not depend on generic parameters
    //~| NOTE its type must not depend on the const parameter `Q`
    //~| NOTE the const parameter `Q` is defined here
    //~| NOTE `K` has type `&'r [Self; Q]`
    //~| NOTE its type must not depend on the lifetime parameter `'r`
    //~| NOTE `K` has type `&'r [Self; Q]`
    //~| ERROR the type of the associated constant `K` must not depend on `Self`
    //~| NOTE its type must not depend on `Self`
    //~| NOTE `K` has type `&'r [Self; Q]`
    //~| ERROR the type of the associated constant `K` must not depend on generic parameters
    //~| NOTE its type must not depend on the const parameter `Q`
    //~| NOTE the const parameter `Q` is defined here
    //~| NOTE `K` has type `&'r [Self; Q]`
    //~| NOTE duplicate diagnostic emitted due to `-Z deduplicate-diagnostics=no`
    //~| NOTE duplicate diagnostic emitted due to `-Z deduplicate-diagnostics=no`
    //~| NOTE duplicate diagnostic emitted due to `-Z deduplicate-diagnostics=no`
    where
        Self: Sized + 'r;
}

fn main() {}
