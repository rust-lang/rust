//! Checks whether we are properly enforcing recursive const stability for traits and trait calls.
//@ compile-flags: -Znext-solver

#![feature(unstable)] // The feature from the ./auxiliary/staged-api.rs file.
#![feature(local_feature)]
#![feature(const_trait_impl)]
#![feature(staged_api, rustc_attrs)]
#![feature(rustc_allow_const_fn_unstable)]
#![stable(feature = "rust1", since = "1.0.0")]

//@ aux-build: staged-api.rs
extern crate staged_api;

use staged_api::*;

#[stable(feature = "rust1", since = "1.0.0")]
pub struct Foo;

#[stable(feature = "rust1", since = "1.0.0")]
impl const MyTrait for Foo {
    fn func() {}
}

// Const stability has no impact on usage in non-const contexts.
fn non_const_context() {
    Unstable::func();
    Unstable::func2();
    Foo::func();
}

// 1. unstably-const fn/trait
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_unstable(feature = "local_feature", issue = "none")]
// `~const` bound on unstably-const trait
const fn unstable_const_context<T: ~const MyTrait>() {
    // calling associated method on unstably-const trait; remote impl; feature enabled
    Unstable::func();

    // calling associated method on unstably-const trait; remote impl; feature disabled
    Unstable::func2();
    //~^ ERROR: `staged_api::MyTrait2` is not yet stable

    // calling associated method on unstably-const trait; local impl
    Foo::func();

    // calling associated method on unstably-const trait; from bounds
    T::func();
}

#[const_trait]
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_unstable(feature = "local_feature", issue = "none")]
// `~const` supertrait on unstably-const trait
trait LocalUnstable: ~const MyTrait {}

// 2. indirectly stable const fn
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_unstable(feature = "local_feature", issue = "none")]
#[rustc_const_stable_indirect]
// `~const` bound on unstably-const trait
const fn indirectly_stable<T: ~const MyTrait>() {
    //~^ exposed to stable cannot use `#[feature(unstable)]`

    // calling associated method on unstably-const trait; remote impl; feature enabled
    Unstable::func();
    //~^ exposed to stable cannot use `#[feature(const_trait_impl)]`
    //~| exposed to stable cannot use `#[feature(unstable)]`

    // calling associated method on unstably-const trait; remote impl; feature disabled
    Unstable::func2();
    //~^ ERROR: `staged_api::MyTrait2` is not yet stable
    //~| exposed to stable cannot use `#[feature(const_trait_impl)]`

    // calling associated method on unstably-const trait; local impl
    Foo::func();
    //~^ exposed to stable cannot use `#[feature(const_trait_impl)]`
    //~| exposed to stable cannot use `#[feature(unstable)]`
    // calling associated method on unstably-const trait; from bounds

    T::func();
    //~^ exposed to stable cannot use `#[feature(const_trait_impl)]`
    //~| exposed to stable cannot use `#[feature(unstable)]`
}

// 3. implicitly stable const fn/trait
#[stable(feature = "rust1", since = "1.0.0")]
// `~const` bound on unstably-const trait
const fn implicitly_stable<T: ~const MyTrait>() {
    //~^ exposed to stable cannot use `#[feature(unstable)]`

    // calling associated method on unstably-const trait; remote impl; feature enabled
    Unstable::func();
    //~^ exposed to stable cannot use `#[feature(const_trait_impl)]`
    //~| exposed to stable cannot use `#[feature(unstable)]`

    // calling associated method on unstably-const trait; remote impl; feature disabled
    Unstable::func2();
    //~^ ERROR: `staged_api::MyTrait2` is not yet stable
    //~| exposed to stable cannot use `#[feature(const_trait_impl)]`

    // calling associated method on unstably-const trait; local impl
    Foo::func();
    //~^ exposed to stable cannot use `#[feature(const_trait_impl)]`
    //~| exposed to stable cannot use `#[feature(unstable)]`
    // calling associated method on unstably-const trait; from bounds

    T::func();
    //~^ exposed to stable cannot use `#[feature(const_trait_impl)]`
    //~| exposed to stable cannot use `#[feature(unstable)]`
}

#[const_trait]
#[stable(feature = "rust1", since = "1.0.0")]
// `~const` supertrait on unstably-const trait
trait ImplicitlyStable: ~const MyTrait {}
//~^ exposed to stable cannot use `#[feature(unstable)]`

// 4. explicitly stable const fn/trait
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_stable(feature = "rust1", since = "1.0.0")]
// `~const` bound on unstably-const trait
const fn explicitly_stable<T: ~const MyTrait>() {
    //~^ exposed to stable cannot use `#[feature(unstable)]`

    // calling associated method on unstably-const trait; remote impl; feature enabled
    Unstable::func();
    //~^ exposed to stable cannot use `#[feature(const_trait_impl)]`
    //~| exposed to stable cannot use `#[feature(unstable)]`

    // calling associated method on unstably-const trait; remote impl; feature disabled
    Unstable::func2();
    //~^ ERROR: `staged_api::MyTrait2` is not yet stable
    //~| exposed to stable cannot use `#[feature(const_trait_impl)]`

    // calling associated method on unstably-const trait; local impl
    Foo::func();
    //~^ exposed to stable cannot use `#[feature(const_trait_impl)]`
    //~| exposed to stable cannot use `#[feature(unstable)]`
    // calling associated method on unstably-const trait; from bounds

    T::func();
    //~^ exposed to stable cannot use `#[feature(const_trait_impl)]`
    //~| exposed to stable cannot use `#[feature(unstable)]`
}

#[const_trait]
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_stable(feature = "rust1", since = "1.0.0")]
// `~const` supertrait on unstably-const trait
trait ExplicitlyStable: ~const MyTrait {}
//~^ exposed to stable cannot use `#[feature(unstable)]`

fn main() {}
