//@ compile-flags: -Znext-solver=coherence
//@ check-pass
#![feature(rustc_attrs)]
#![rustc_no_implicit_bounds]

// A regression test for trait-system-refactor-initiative#70.

trait Trait {
    type Assoc;
}

struct W<T>(*mut T);
impl<T> Trait for W<W<T>>
where
    W<T>: Trait,
{
    type Assoc = ();
}

trait NoOverlap {}
impl<T: Trait<Assoc = u32>> NoOverlap for T {}
// `Projection(<W<_> as Trait>::Assoc, u32)` should result in error even
// though applying the impl results in overflow. This is necessary to match
// the behavior of the old solver.
impl<T> NoOverlap for W<T> {}

fn main() {}
