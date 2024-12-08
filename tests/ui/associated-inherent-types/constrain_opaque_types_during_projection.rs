//@ check-pass

#![feature(inherent_associated_types, type_alias_impl_trait)]
#![allow(incomplete_features)]

struct Foo<T>(T);

impl Foo<i32> {
    type Assoc = u32;
}

type Tait = impl Sized;

fn bar(_: Tait) {
    let x: Foo<Tait>::Assoc = 42;
}

fn main() {}
