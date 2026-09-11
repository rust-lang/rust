//! Regression test for <https://github.com/rust-lang/rust/issues/162331>
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ check-pass

#![feature(const_trait_impl)]
#![feature(const_clone)]

// At the time of writing, when resolving an instance for `<u8 as Trait>::N`, the environment
// contains `f`'s `u8: Trait` clause. The old solver dropped the where clause candidate in favor of
// the `impl Trait for u8` candidate, but the new solver didn't, which resulted in ambiguity.

pub trait Trait {
    const N: usize;
}

impl Trait for u8 {
    const N: usize = 0;
}

pub fn f()
where
    u8: Trait,
{
    match 0 {
        <u8 as Trait>::N => {}
        _ => {}
    }
}

// At the time of writing, `ZERO` is evaluated in an environment containing `g`'s `(u8,): Clone`
// clause. This wasn't dropped in favor of the built-in `(u8,): Clone` impl when resolving an
// instance for `<(u8,) as Clone>::clone`, which resulted in ambiguity.

const ZERO: (u8,) = (0,).clone();

fn g()
where
    (u8,): Clone
{
    match (0,) {
        ZERO => {}
        _ => {}
    }
}

fn main() {}
