//! Checks whether we are properly enforcing recursive const stability for trait calls.
//@ compile-flags: -Znext-solver

#![feature(unstable)] // The feature from the ./auxiliary/staged-api.rs file.
#![feature(local_feature)]
#![feature(const_trait_impl)]
#![feature(staged_api)]
#![feature(rustc_allow_const_fn_unstable)]
#![stable(feature = "rust1", since = "1.0.0")]

//@ aux-build: staged-api.rs
extern crate staged_api;

use staged_api::*;

#[stable(feature = "rust1", since = "1.0.0")]
pub struct Foo;

#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_unstable(feature = "local_feature", issue = "none")]
impl const MyTrait for Foo {
    fn func() {}
}

#[rustc_allow_const_fn_unstable(const_trait_impl)]
const fn conditionally_const<T: ~const MyTrait>() {
    T::func();
}

// Const stability has no impact on usage in non-const contexts.
fn non_const_context() {
    Unstable::func();
    Foo::func();
}

#[unstable(feature = "none", issue = "none")]
const fn const_context() {
    Unstable::func();
    //~^ ERROR cannot use `#[feature(const_trait_impl)]`
    Foo::func();
    //~^ ERROR cannot use `#[feature(const_trait_impl)]`
    Unstable2::func();
    //~^ ERROR cannot use `#[feature(const_trait_impl)]`
    conditionally_const::<Foo>();
    //~^ ERROR cannot use `#[feature(const_trait_impl)]`
}

#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_unstable(feature = "local_feature", issue = "none")]
pub const fn const_context_not_const_stable() {
    Unstable::func();
    Foo::func();
    Unstable2::func();
    conditionally_const::<Foo>();
}

#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_stable(feature = "cheese", since = "1.0.0")]
const fn stable_const_context() {
    Unstable::func();
    //~^ ERROR cannot use `#[feature(const_trait_impl)]`
    Foo::func();
    //~^ ERROR cannot use `#[feature(const_trait_impl)]`
    const_context_not_const_stable();
    //~^ ERROR cannot use `#[feature(local_feature)]`
    conditionally_const::<Foo>();
    //~^ ERROR cannot use `#[feature(const_trait_impl)]`
}

fn main() {}
