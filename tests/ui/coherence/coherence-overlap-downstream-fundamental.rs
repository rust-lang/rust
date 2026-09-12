//! Regression test for <https://github.com/rust-lang/rust/issues/43355>.
//! Test trait relationship defined as `Trait1<X> for T where T: Trait2<X>`
//! rejects implementations of `Trait1` with generics over `#[fundamental]`
//! types, as a downstream crate can implement dependent `Trait2` for the same
//! type with the same generics, causing coherence breakage.
//!
//! This used to ICE if downstream crate tried to `impl Trait2<Box<_>> for A`.
//@ dont-require-annotations: NOTE

pub trait Trait1<X> {
    type Output;
}

pub trait Trait2<X> {}

pub struct A;

impl<X, T> Trait1<X> for T where T: Trait2<X> {
    type Output = ();
}

impl<X> Trait1<Box<X>> for A {
//~^ ERROR conflicting implementations of trait
//~| NOTE downstream crates may implement trait `Trait2<std::boxed::Box<_>>` for type `A`
    type Output = i32;
}

fn main() {}
