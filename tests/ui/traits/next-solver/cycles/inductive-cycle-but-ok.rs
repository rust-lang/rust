//@ compile-flags: -Znext-solver
//@ check-pass
#![feature(trivial_bounds, marker_trait_attr)]
#![allow(trivial_bounds)]

// This previously triggered a bug in the provisional cache.
//
// This has the proof tree
// - `Root: Trait` proven via impl
//   - `MultipleCandidates: Trait`
//     - candidate: overflow-impl
//       - `Root: Trait` (inductive cycle ~> OVERFLOW)
//     - candidate: trivial-impl ~> YES
//     - merge respones ~> YES
//   - `MultipleCandidates: Trait` (in provisional cache ~> OVERFLOW)
//
// We previously incorrectly treated the `MultipleCandidates: Trait` as
// overflow because it was in the cache and reached via an inductive cycle.
// It should be `YES`.

struct Root;
struct MultipleCandidates;

#[marker]
trait Trait {}
impl Trait for Root
where
    MultipleCandidates: Trait,
    MultipleCandidates: Trait,
{}

// overflow-impl
impl Trait for MultipleCandidates
where
    Root: Trait,
{}
// trivial-impl
impl Trait for MultipleCandidates {}

fn impls_trait<T: Trait>() {}

fn main() {
    impls_trait::<Root>();
}
