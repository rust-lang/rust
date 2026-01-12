//! Regression test for <https://github.com/rust-lang/rust/issues/150354>
//@ edition 2024

#![allow(incomplete_features)]
#![feature(min_generic_const_args, adt_const_params)]

#[derive(Eq, PartialEq, core::marker::ConstParamTy)]
struct Foo;

trait Trait {
    #[type_const]
    const ASSOC: u32;
}

fn foo<const N: Foo>() {}

fn bar<T, const N: u32>() {
    foo::<{ Option::Some::<u32> { 0: N } }>;
    //~^ ERROR the constant `Option::<u32>::Some(N)` is not of type `Foo`
}

fn baz<T: Trait>() {
    foo::<{ Option::Some::<u32> { 0: <T as Trait>::ASSOC } }>();
    //~^ ERROR the constant `Option::<u32>::Some(<T as Trait>::ASSOC)` is not of type `Foo`
}

fn main() {}

fn test_ice_missing_bound<T>() {
    foo::<{ Option::Some::<u32> { 0: <T as Trait>::ASSOC } }>();
    //~^ ERROR the trait bound `T: Trait` is not satisfied
    //~| ERROR the constant `Option::<u32>::Some(_)` is not of type `Foo`
}

fn test_underscore_inference() {
    foo::<{ Option::Some::<u32> { 0: _ } }>();
    //~^ ERROR the constant `Option::<u32>::Some(_)` is not of type `Foo`
}
