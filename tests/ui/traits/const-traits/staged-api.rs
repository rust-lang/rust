//@ revisions: stable unstable
//@ compile-flags: -Znext-solver

#![cfg_attr(unstable, feature(unstable))] // The feature from the ./auxiliary/staged-api.rs file.
#![cfg_attr(unstable, feature(local_feature))]
#![feature(const_trait_impl, effects)]
#![allow(incomplete_features)]
#![feature(staged_api)]
#![stable(feature = "rust1", since = "1.0.0")]

//@ aux-build: staged-api.rs
extern crate staged_api;

use staged_api::*;

#[stable(feature = "rust1", since = "1.0.0")]
pub struct Foo;

#[stable(feature = "rust1", since = "1.0.0")]
#[cfg_attr(unstable, rustc_const_unstable(feature = "local_feature", issue = "none"))]
#[cfg_attr(stable, rustc_const_stable(feature = "local_feature", since = "1.0.0"))]
impl const MyTrait for Foo {
    //[stable]~^ ERROR trait implementations cannot be const stable yet
    fn func() {}
}

// Const stability has no impact on usage in non-const contexts.
fn non_const_context() {
    Unstable::func();
    Foo::func();
}

#[unstable(feature = "none", issue = "none")]
const fn const_context() {
    Unstable::func();
    //[unstable]~^ ERROR cannot use `#[feature(unstable)]`
    //[stable]~^^ ERROR not yet stable as a const fn
    Foo::func();
    //[unstable]~^ ERROR cannot use `#[feature(local_feature)]`
    //[stable]~^^ cannot be (indirectly) exposed to stable
    // We get the error on `stable` since this is a trait function.
    Unstable2::func();
    //~^ ERROR not yet stable as a const fn
    // ^ fails, because the `unstable2` feature is not active
}

#[stable(feature = "rust1", since = "1.0.0")]
#[cfg_attr(unstable, rustc_const_unstable(feature = "local_feature", issue = "none"))]
pub const fn const_context_not_const_stable() {
    //[stable]~^ ERROR function has missing const stability attribute
    Unstable::func();
    //[stable]~^ ERROR not yet stable as a const fn
    Foo::func();
    //[stable]~^ cannot be (indirectly) exposed to stable
    // We get the error on `stable` since this is a trait function.
    Unstable2::func();
    //~^ ERROR not yet stable as a const fn
    // ^ fails, because the `unstable2` feature is not active
}

#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_stable(feature = "cheese", since = "1.0.0")]
const fn stable_const_context() {
    Unstable::func();
    //[unstable]~^ ERROR cannot use `#[feature(unstable)]`
    //[stable]~^^ ERROR not yet stable as a const fn
    Foo::func();
    //[unstable]~^ ERROR cannot use `#[feature(local_feature)]`
    //[stable]~^^ cannot be (indirectly) exposed to stable
    // We get the error on `stable` since this is a trait function.
    const_context_not_const_stable()
    //[unstable]~^ ERROR cannot use `#[feature(local_feature)]`
}

fn main() {}
