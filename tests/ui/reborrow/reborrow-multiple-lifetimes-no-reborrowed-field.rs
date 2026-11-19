//! Regression test for <https://github.com/rust-lang/rust/issues/165653>.
//!
//! A `Reborrow` type with no `&mut T` or non-`Copy` `Reborrow` field is borrowed
//! as a whole under its single lifetime, so it must not have more than one.

#![feature(reborrow)]

use std::marker::{PhantomData, Reborrow};

struct SharedOnly<'a, 'b> {
    a: &'a &'b str,
}

impl<'a, 'b> Reborrow for SharedOnly<'a, 'b> {}
//~^ ERROR implementing `Reborrow` on a type with 2 lifetime arguments requires at least one field to be reborrowed

// The body from the issue. It never reaches borrowck, because coherence rejects the impl first.
impl<'a, 'b> SharedOnly<'a, 'b> {
    fn clone(&self) -> Self {
        *self
    }
}

struct PhantomOnly<'a, 'b>(PhantomData<(&'a (), &'b ())>);

impl<'a, 'b> Reborrow for PhantomOnly<'a, 'b> {}
//~^ ERROR implementing `Reborrow` on a type with 2 lifetime arguments requires at least one field to be reborrowed

// Borrowck does not reborrow type parameter fields, even if they implement `Reborrow`.
struct ParamOnly<'a, 'b, T: Reborrow> {
    t: T,
    marker: PhantomData<(&'a (), &'b ())>,
}

impl<'a, 'b, T: Reborrow> Reborrow for ParamOnly<'a, 'b, T> {}
//~^ ERROR implementing `Reborrow` on a type with 2 lifetime arguments requires at least one field to be reborrowed

trait HasAssoc<'a> {
    type Assoc;
}

impl<'a> HasAssoc<'a> for () {
    type Assoc = &'a mut ();
}

// Borrowck looks at the declared field type, so this field is not reborrowed even
// though it normalizes to `&mut ()`.
struct AssocOnly<'a, 'b> {
    assoc: <() as HasAssoc<'a>>::Assoc,
    shared: &'b (),
}

impl<'a, 'b> Reborrow for AssocOnly<'a, 'b> {}
//~^ ERROR implementing `Reborrow` on a type with 2 lifetime arguments requires at least one field to be reborrowed

fn main() {}
