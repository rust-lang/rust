//@ check-pass
//! A `Reborrow` impl can make a field `Copy` with a where clause even though the
//! field's type has no `Copy` impl and is not `Reborrow`. That field is copied, so
//! borrowck must not walk into it as if it were a `Reborrow` type.

#![feature(reborrow, trivial_bounds)]
#![allow(trivial_bounds)]

use std::marker::Reborrow;

struct Inner<'x, 'y>(&'x (), &'y ());

struct WithMutField<'a, 'b> {
    inner: Inner<'a, 'b>,
    mutable: &'a mut (),
}

impl<'a, 'b> Reborrow for WithMutField<'a, 'b> where Inner<'a, 'b>: Copy {}

fn use_it<'a, 'b>(x: WithMutField<'a, 'b>) -> WithMutField<'a, 'b>
where
    Inner<'a, 'b>: Copy,
{
    x
}

fn main() {}
