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

#[rustc_allow_const_fn_unstable(const_trait_impl, unstable)]
const fn conditionally_const<T: [const] MyTrait>() {
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
    //~| ERROR cannot use `#[feature(unstable)]`
    Foo::func();
    //~^ ERROR cannot use `#[feature(const_trait_impl)]`
    //~| ERROR cannot use `#[feature(unstable)]`
    Unstable2::func();
    //~^ ERROR cannot use `#[feature(const_trait_impl)]`
    //~| ERROR cannot use `#[feature(unstable)]`
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
    //~| ERROR cannot use `#[feature(unstable)]`
    Foo::func();
    //~^ ERROR cannot use `#[feature(const_trait_impl)]`
    //~| ERROR cannot use `#[feature(unstable)]`
    const_context_not_const_stable();
    //~^ ERROR cannot use `#[feature(local_feature)]`
    conditionally_const::<Foo>();
    //~^ ERROR cannot use `#[feature(const_trait_impl)]`
}

const fn implicitly_stable_const_context() {
    Unstable::func();
    //~^ ERROR cannot use `#[feature(const_trait_impl)]`
    //~| ERROR cannot use `#[feature(unstable)]`
    Foo::func();
    //~^ ERROR cannot use `#[feature(const_trait_impl)]`
    //~| ERROR cannot use `#[feature(unstable)]`
    const_context_not_const_stable();
    //~^ ERROR cannot use `#[feature(local_feature)]`
    conditionally_const::<Foo>();
    //~^ ERROR cannot use `#[feature(const_trait_impl)]`
}

// check that const stability of impls and traits must match
#[const_trait]
#[rustc_const_unstable(feature = "beef", issue = "none")]
trait U {}

#[const_trait]
#[rustc_const_stable(since = "0.0.0", feature = "beef2")]
trait S {}

// implied stable
impl const U for u8 {}
//~^ ERROR const stability on the impl does not match the const stability on the trait

#[rustc_const_stable(since = "0.0.0", feature = "beef2")]
impl const U for u16 {}
//~^ ERROR const stability on the impl does not match the const stability on the trait
//~| ERROR trait implementations cannot be const stable yet

#[rustc_const_unstable(feature = "beef", issue = "none")]
impl const U for u32 {}

// implied stable
impl const S for u8 {}

#[rustc_const_stable(since = "0.0.0", feature = "beef2")]
impl const S for u16 {}
//~^ ERROR trait implementations cannot be const stable yet

#[rustc_const_unstable(feature = "beef", issue = "none")]
impl const S for u32 {}
//~^ ERROR const stability on the impl does not match the const stability on the trait

fn main() {}
