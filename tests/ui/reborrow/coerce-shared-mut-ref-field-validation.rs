//@ check-pass

//! Test that CoerceShared can resolve field types through aliases and GATs.
//! Also test that reference shared coercing does not produce invalid lifetime relations.

#![feature(reborrow)]

use std::marker::{CoerceShared, Reborrow};

trait Trait {
    type Assoc;
}

impl Trait for i32 {
    type Assoc = u32;
}

type AliasMutField<'a> = &'a mut <i32 as Trait>::Assoc;
type AliasRefField<'a> = &'a u32;

struct AliasMut<'a> {
    value: AliasMutField<'a>,
}

impl Reborrow for AliasMut<'_> {}

#[derive(Copy, Clone)]
struct AliasRef<'a> {
    value: AliasRefField<'a>,
}

impl<'a> CoerceShared<AliasRef<'a>> for AliasMut<'a> {}

struct InnerLifetimeMut<'a> {
    value: &'a mut &'static (),
}

impl Reborrow for InnerLifetimeMut<'_> {}

#[derive(Copy, Clone)]
struct InnerLifetimeRef<'a> {
    value: &'a &'a (),
}

// No error explicitly necessary: &'a mut &'static T -> &'a &'a T is a valid coercion. We might
// still want to error on it because it's mostly meaningless, though.
impl<'a> CoerceShared<InnerLifetimeRef<'a>> for InnerLifetimeMut<'a> {}

fn main() {}
