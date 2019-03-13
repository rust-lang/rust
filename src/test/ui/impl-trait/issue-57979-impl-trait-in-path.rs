// rust-lang/rust#57979 : the initial support for `impl Trait` didn't
// properly check syntax hidden behind an associated type projection.
// Here we test behavior of occurrences of `impl Trait` within a path
// component in that context.

mod allowed {
    #![allow(nested_impl_trait)]

    pub trait Bar { }
    pub trait Quux<T> { type Assoc; }
    pub fn demo(_: impl Quux<(), Assoc=<() as Quux<impl Bar>>::Assoc>) { }
    impl<T> Quux<T> for () { type Assoc = u32; }
}

mod warned {
    #![warn(nested_impl_trait)]

    pub trait Bar { }
    pub trait Quux<T> { type Assoc; }
    pub fn demo(_: impl Quux<(), Assoc=<() as Quux<impl Bar>>::Assoc>) { }
    //~^ WARN `impl Trait` is not allowed in path parameters
    //~| WARN will become a hard error in a future release!
    impl<T> Quux<T> for () { type Assoc = u32; }
}

mod denied {
    #![deny(nested_impl_trait)]

    pub trait Bar { }
    pub trait Quux<T> { type Assoc; }
    pub fn demo(_: impl Quux<(), Assoc=<() as Quux<impl Bar>>::Assoc>) { }
    //~^ ERROR `impl Trait` is not allowed in path parameters
    //~| WARN will become a hard error in a future release!
    impl<T> Quux<T> for () { type Assoc = u32; }
}

fn main() { }
