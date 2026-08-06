//@ check-pass

// Previously we didn't normalize the source type of `CoerceShared` when
// checking its trait impls.
// We also didn't normalize the types of fields in the wrappers.
// So they fail the trait impl check in `coerce_shared_info` even if
// the normalized types satisfy the requirements.

#![feature(reborrow)]
use std::marker::{CoerceShared, Reborrow};

struct NonParam;

trait HasAssoc<'a> {
    type Assoc;
}

struct A;
impl<'a> HasAssoc<'a> for A {
    type Assoc = &'a mut NonParam;
}
type MutAlias<'a> = <A as HasAssoc<'a>>::Assoc;

struct B;
impl<'a> HasAssoc<'a> for B {
    type Assoc = &'a NonParam;
}
type RefAlias<'a> = <B as HasAssoc<'a>>::Assoc;

struct CustomMut<'a>(MutAlias<'a>);
struct CustomRef<'a>(RefAlias<'a>);

struct C;
impl<'a> HasAssoc<'a> for C {
    type Assoc = CustomMut<'a>;
}
type CustomMutAlias<'a> = <C as HasAssoc<'a>>::Assoc;

struct D;
impl<'a> HasAssoc<'a> for D {
    type Assoc = CustomRef<'a>;
}
type CustomRefAlias<'a> = <D as HasAssoc<'a>>::Assoc;

impl<'a> Clone for CustomRef<'a> {
    fn clone(&self) -> Self {
        Self(self.0)
    }
}
impl<'a> Copy for CustomRef<'a> {}


impl<'a> Reborrow for CustomMut<'a> {}
impl<'a> CoerceShared<CustomRefAlias<'a>> for CustomMutAlias<'a> {}

fn main() {}
