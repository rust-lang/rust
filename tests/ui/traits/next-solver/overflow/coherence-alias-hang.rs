//@ check-pass
//@ revisions: ai ia ii

// Regression test for nalgebra hang <https://github.com/rust-lang/rust/issues/130056>.

#![feature(lazy_type_alias)]
#![feature(rustc_attrs)]
#![rustc_no_implicit_bounds]
#![allow(incomplete_features)]

type Id<T> = T;
trait NotImplemented {}

struct W<T, U>(*const T, *const U);
trait Trait {
    type Assoc;
}
impl<T: Trait> Trait for W<T, T> {
    #[cfg(ai)]
    type Assoc = W<T::Assoc, Id<T::Assoc>>;
    #[cfg(ia)]
    type Assoc = W<Id<T::Assoc>, T::Assoc>;
    #[cfg(ii)]
    type Assoc = W<Id<T::Assoc>, Id<T::Assoc>>;
}

trait Overlap<T> {}
impl<T> Overlap<T> for W<T, T> {}
impl<T: Trait + NotImplemented> Overlap<T::Assoc> for T {}

fn main() {}
