//@ check-pass
//! This is a reproducer for the ICE 6840: https://github.com/rust-lang/rust-clippy/issues/6840.
//! The ICE is caused by `TyCtxt::layout_of` and `is_normalizable` not being strict enough
#![allow(dead_code)]
#![deny(clippy::zero_sized_map_values)] // For ICE 14822
use std::collections::HashMap;

pub trait Rule {
    type DependencyKey;
}

pub struct RuleEdges<R: Rule> {
    dependencies: R::DependencyKey,
}

type RuleDependencyEdges<R> = HashMap<u32, RuleEdges<R>>;

// reproducer from the GitHub issue ends here
//   but check some additional variants
type RuleDependencyEdgesArray<R> = HashMap<u32, [RuleEdges<R>; 8]>;
type RuleDependencyEdgesSlice<R> = HashMap<u32, &'static [RuleEdges<R>]>;
type RuleDependencyEdgesRef<R> = HashMap<u32, &'static RuleEdges<R>>;
type RuleDependencyEdgesRaw<R> = HashMap<u32, *const RuleEdges<R>>;
type RuleDependencyEdgesTuple<R> = HashMap<u32, (RuleEdges<R>, RuleEdges<R>)>;

// and an additional checks to make sure fix doesn't have stack-overflow issue
//   on self-containing types
pub struct SelfContaining {
    inner: Box<SelfContaining>,
}
type SelfContainingEdges = HashMap<u32, SelfContaining>;

fn main() {}
