//@ check-pass

#![feature(reborrow)]
use std::marker::{CoerceShared, Reborrow};

trait HasAssoc<'a, T> {
    type Assoc;
}

struct Mut;
impl<'a, T> HasAssoc<'a, T> for Mut where T: 'a {
    type Assoc = CustomMut<'a, T>;
}
type CustomMutAlias<'a, T> = <Mut as HasAssoc<'a, T>>::Assoc;

struct A;
impl<'a, T: 'a> HasAssoc<'a, T> for A {
    type Assoc = &'a mut T;
}
type MutAlias<'a, T> = <A as HasAssoc<'a, T>>::Assoc;

struct B;
impl<'a, T: 'a> HasAssoc<'a, T> for B {
    type Assoc = &'a T;
}
type RefAlias<'a, T> = <B as HasAssoc<'a, T>>::Assoc;

struct CustomMut<'a, T: 'a>(MutAlias<'a, T>);
struct CustomRef<'a, T: 'a>(RefAlias<'a, T>);

impl<'a, T> Clone for CustomRef<'a, T> {
    fn clone(&self) -> Self {
        Self(self.0)
    }
}
impl<'a, T> Copy for CustomRef<'a, T> {}


impl<'a, T> Reborrow for CustomMut<'a, T> {}
impl<'a, T: 'a> CoerceShared<CustomRef<'a, T>> for CustomMutAlias<'a, T> {}
//~^ WARN: type parameter `T` must be covered by another type when it appears before the first local type
//~| WARN: this was previously accepted by the compiler but is being phased out


fn main() {}
