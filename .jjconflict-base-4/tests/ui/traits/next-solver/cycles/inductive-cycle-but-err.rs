//@ compile-flags: -Znext-solver
#![feature(trivial_bounds, marker_trait_attr)]
#![allow(trivial_bounds)]
// This previously triggered a bug in the provisional cache.
//
// This has the proof tree
// - `MultipleCandidates: Trait` proven via impl-one
//   - `MultipleNested: Trait` via impl
//     - `MultipleCandidates: Trait` (inductive cycle ~> OVERFLOW)
//     - `DoesNotImpl: Trait` (ERR)
// - `MultipleCandidates: Trait` proven via impl-two
//   - `MultipleNested: Trait` (in provisional cache ~> OVERFLOW)
//
// We previously incorrectly treated the `MultipleCandidates: Trait` as
// overflow because it was in the cache and reached via an inductive cycle.
// It should be `NoSolution`.

struct MultipleCandidates;
struct MultipleNested;
struct DoesNotImpl;

#[marker]
trait Trait {}

// impl-one
impl Trait for MultipleCandidates
where
    MultipleNested: Trait
{}

// impl-two
impl Trait for MultipleCandidates
where
    MultipleNested: Trait,
{}

// We ignore the trivially true global where-bounds when checking that this
// impl is well-formed, meaning that we depend on `MultipleNested: Trait` when
// recursively proving `MultipleCandidates: Trait`.
//
// These overflow errors will disappear once we treat these cycles as either
// productive or an error.
impl Trait for MultipleNested
//~^ ERROR overflow evaluating the requirement `MultipleNested: Trait`
where
    MultipleCandidates: Trait,
    //~^ ERROR overflow evaluating the requirement `MultipleCandidates: Trait`
    DoesNotImpl: Trait,
{}

fn impls_trait<T: Trait>() {}

fn main() {
    impls_trait::<MultipleCandidates>();
    //~^ ERROR the trait bound `MultipleCandidates: Trait` is not satisfied
}
