// rust-lang/rust#57979 : the initial support for `impl Trait` didn't
// properly check syntax hidden behind an associated type projection,
// but it did catch *some cases*. This is checking that we continue to
// properly emit errors for those, even with the new
// future-incompatibility warnings.
//
// issue-57979-nested-impl-trait-in-assoc-proj.rs shows the main case
// that we were previously failing to catch.

struct Deeper<T>(T);

mod allowed {
    #![allow(nested_impl_trait)]

    pub trait Foo<T> { }
    pub trait Bar { }
    pub trait Quux { type Assoc; }
    pub fn demo(_: impl Quux<Assoc=super::Deeper<impl Foo<impl Bar>>>) { }
    //~^ ERROR nested `impl Trait` is not allowed
}

mod warned {
    #![warn(nested_impl_trait)]

    pub trait Foo<T> { }
    pub trait Bar { }
    pub trait Quux { type Assoc; }
    pub fn demo(_: impl Quux<Assoc=super::Deeper<impl Foo<impl Bar>>>) { }
    //~^ ERROR nested `impl Trait` is not allowed
}

mod denied {
    #![deny(nested_impl_trait)]

    pub trait Foo<T> { }
    pub trait Bar { }
    pub trait Quux { type Assoc; }
    pub fn demo(_: impl Quux<Assoc=super::Deeper<impl Foo<impl Bar>>>) { }
    //~^ ERROR nested `impl Trait` is not allowed
}

fn main() { }
