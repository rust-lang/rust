//@ check-pass
// See issue #109356. We don't want a false positive to the `implied_bounds_entailment` lint.

use std::borrow::Cow;

pub trait Trait {
    fn method(self) -> Option<Cow<'static, str>>
    where
        Self: Sized;
}

impl<'a> Trait for Cow<'a, str> {
    // If we're not careful here, we'll check `WF(return-type)` using the trait
    // and impl where clauses, requiring that `Cow<'a, str>: Sized`. This is
    // obviously true, but if we pick the `Self: Sized` clause from the trait
    // over the "inherent impl", we will require `'a == 'static`, which triggers
    // the `implied_bounds_entailment` lint.
    fn method(self) -> Option<Cow<'static, str>> {
        None
    }
}

fn main() {}
