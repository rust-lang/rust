//@ check-pass
//@ revisions: ai ia ii
//@ compile-flags: -Znext-solver=coherence

// Regression test for nalgebra hang <https://github.com/rust-lang/rust/issues/130056>.

#![feature(lazy_type_alias)]
#![allow(incomplete_features)]

type Id<T: ?Sized> = T;
trait NotImplemented {}

struct W<'a, T: ?Sized, U: ?Sized>(&'a (), *const T, *const U);
trait Trait {
    type Assoc: ?Sized;
}
impl<'a, T: ?Sized + Trait> Trait for W<'a, T, T> {
    #[cfg(ai)]
    type Assoc = W<'a, T::Assoc, Id<T::Assoc>>;
    #[cfg(ia)]
    type Assoc = W<'a, Id<T::Assoc>, T::Assoc>;
    #[cfg(ii)]
    type Assoc = W<'a, Id<T::Assoc>, Id<T::Assoc>>;
}

trait Overlap<T: ?Sized> {}
impl<'a, T: ?Sized> Overlap<T> for W<'a, T, T> {}
impl<T: ?Sized + Trait + NotImplemented> Overlap<T::Assoc> for T {}

fn main() {}
