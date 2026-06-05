//@ normalize-stderr: "\n\n\z" -> "\n"

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
    //~^ ERROR
}

impl<'a> CoerceShared<InnerLifetimeRef<'a>> for InnerLifetimeMut<'a> {}

fn main() {}
