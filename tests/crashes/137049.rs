//@ known-bug: #137049
//@ compile-flags: --crate-type=lib
#![feature(type_alias_impl_trait)]

use std::marker::PhantomData;

trait Project1 {
    type Assoc1;
}

impl<T> Project1 for T {
    type Assoc1 = ();
}

trait Project2 {
    type Assoc2;
}

impl<T: Project1<Assoc1 = ()>> Project2 for PhantomData<T> {
    type Assoc2 = ();
}

type Alias<T> = impl Project2;

#[define_opaque(Alias)]
fn constrain<T>() -> Alias<T> {
    PhantomData::<T>
}

struct AdtConstructor<T: Project1>(<Alias<T> as Project2>::Assoc2);
