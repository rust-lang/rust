//@ compile-flags: -Znext-solver=globally

#![feature(inherent_associated_types)]
#![feature(type_alias_impl_trait)]
#![allow(incomplete_features)]

struct Foo<T>(T);

impl Foo<Vec<[u32]>> {
    //~^ ERROR the size for values of type `[u32]` cannot be known at compilation time
    type Assoc = u32;
}

type Tait = impl Sized;

#[define_opaque(Tait)]
fn bar() {
    let _: Foo<Tait>::Assoc = 42;
    //~^ ERROR the associated type `Assoc` exists for `Foo<Tait>`, but its trait bounds were not satisfied
}

fn main() {}
