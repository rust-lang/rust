// rust-lang/rust#57979 : the initial support for `impl Trait` didn't
// properly check syntax hidden behind an associated type projection.
// Here we test behavior of occurrences of `impl Trait` within an
// `impl Trait` in that context.

mod allowed {
    #![allow(nested_impl_trait)]

    pub trait Foo<T> { }
    pub trait Bar { }
    pub trait Quux { type Assoc; }
    pub fn demo(_: impl Quux<Assoc=impl Foo<impl Bar>>) { }
}

mod warned {
    #![warn(nested_impl_trait)]

    pub trait Foo<T> { }
    pub trait Bar { }
    pub trait Quux { type Assoc; }
    pub fn demo(_: impl Quux<Assoc=impl Foo<impl Bar>>) { }
    //~^ WARN nested `impl Trait` is not allowed
    //~| WARN will become a hard error in a future release!
}

mod denied {
    #![deny(nested_impl_trait)]

    pub trait Foo<T> { }
    pub trait Bar { }
    pub trait Quux { type Assoc; }
    pub fn demo(_: impl Quux<Assoc=impl Foo<impl Bar>>) { }
    //~^ ERROR nested `impl Trait` is not allowed
    //~| WARN will become a hard error in a future release!
}

fn main() { }
