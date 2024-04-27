//@ check-pass

// This test verifies that negative trait predicate cannot be satisfied from a
// positive param-env candidate.

// Negative coherence is one of the only places where we actually construct and
// evaluate negative predicates. Specifically, when verifying whether the first
// and second impls below overlap, we do not want to consider them disjoint,
// otherwise the second impl would be missing an associated type `type Item`
// which is provided by the first impl that it is specializing.

#![feature(specialization)]
//~^ WARN the feature `specialization` is incomplete
#![feature(with_negative_coherence)]

trait BoxIter {
    type Item;

    fn last(self) -> Option<Self::Item>;
}

impl<I: Iterator + ?Sized> BoxIter for Box<I> {
    type Item = I::Item;

    default fn last(self) -> Option<I::Item> {
        todo!()
    }
}

// When checking that this impl does/doesn't overlap the one above, we evaluate
// a negative version of all of the where-clause predicates of the impl below.
// For `I: !Iterator`, we should make sure that the param-env clause `I: Iterator`
// from above doesn't satisfy this predicate.
impl<I: Iterator> BoxIter for Box<I> {
    fn last(self) -> Option<I::Item> {
        (*self).last()
    }
}

fn main() {}
