//! Regression test for <https://github.com/rust-lang/rust/issues/106832>.
//!
//! Proving `T::Assoc<_>: Sized` while a where-clause bound `T::Assoc<u8>: Sized` is
//! in scope used to over-eagerly infer the otherwise-unconstrained argument to `u8`,
//! producing a spurious "mismatched types" error. This should compile.

//@ check-pass

#![allow(dead_code)]

trait Trait {
    type Assoc<A>;
}

fn test<T: Trait>()
where
    T::Assoc<u8>: Sized,
{
    // `_` must be inferred from the `1i32` argument, not eagerly unified with `u8`
    // just because `T::Assoc<u8>: Sized` happens to be in the environment.
    constrain::<T, _>(1i32);
}

fn constrain<T: Trait, A>(_: A)
where
    T::Assoc<A>: Sized,
{
}

fn main() {}
