//@ check-pass
//@ revisions: ai ia ii
//@ compile-flags: -Znext-solver=coherence

// Regression test for nalgebra hang <https://github.com/rust-lang/rust/issues/130056>.

#![feature(lazy_type_alias)]
#![feature(rustc_attrs)]
#![rustc_no_implicit_bounds]
#![allow(incomplete_features)]

type Id<T> = T;
trait NotImplemented {}

struct W<'a, T, U>(&'a (), *const T, *const U);
trait Trait {
    type Assoc;
}
impl<'a, T: Trait> Trait for W<'a, T, T> {
    #[cfg(ai)]
    type Assoc = W<'a, T::Assoc, Id<T::Assoc>>;
    #[cfg(ia)]
    type Assoc = W<'a, Id<T::Assoc>, T::Assoc>;
    #[cfg(ii)]
    type Assoc = W<'a, Id<T::Assoc>, Id<T::Assoc>>;
}

trait Overlap<T> {}
impl<'a, T> Overlap<T> for W<'a, T, T> {}
impl<T: Trait + NotImplemented> Overlap<T::Assoc> for T {}

fn main() {}
