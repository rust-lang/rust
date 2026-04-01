#![crate_type = "lib"]

// This test is an example of a filtering lending iterator with GATs from #92985 (that is similar to
// NLL problem case #3) to ensure it "works" with the polonius alpha analysis as with the datalog
// implementation.
//
// The polonius analysis only changes how the `Filter::next` function is borrowcked, not the bounds
// on the predicate from using the GAT. So even if the #92985 limitation is removed, the unrelated
// 'static limitation on the predicate argument is still there, and the pattern is still impractical
// to use in the real world.

//@ ignore-compare-mode-polonius (explicit revisions)
//@ revisions: nll polonius legacy
//@ [nll] known-bug: #92985
//@ [polonius] check-pass
//@ [polonius] compile-flags: -Z polonius=next
//@ [legacy] check-pass
//@ [legacy] compile-flags: -Z polonius=legacy

trait LendingIterator {
    type Item<'a>
    where
        Self: 'a;
    fn next(&mut self) -> Option<Self::Item<'_>>;

    fn filter<P>(self, predicate: P) -> Filter<Self, P>
    where
        Self: Sized,
        P: FnMut(&Self::Item<'_>) -> bool,
    {
        Filter { iter: self, predicate }
    }
}

pub struct Filter<I, P> {
    iter: I,
    predicate: P,
}
impl<I: LendingIterator, P> LendingIterator for Filter<I, P>
where
    P: FnMut(&I::Item<'_>) -> bool,
{
    type Item<'a>
        = I::Item<'a>
    where
        Self: 'a;

    fn next(&mut self) -> Option<I::Item<'_>> {
        while let Some(item) = self.iter.next() {
            if (self.predicate)(&item) {
                return Some(item);
            }
        }
        return None;
    }
}
