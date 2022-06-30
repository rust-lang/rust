// known-bug
// check-pass

#![feature(generic_associated_types)]

trait LendingIterator {
    type Item<'a>
    where
        Self: 'a;
}

impl<'slice> LendingIterator for &'slice [u32] {
    type Item<'a> = &'a [u32]
    where
        Self: 'a;
}

impl<'slice> LendingIterator for [u32] {
    type Item<'a> = &'a [u32]
    where
        Self: 'a;
}

fn broke<T: ?Sized>() -> Option<&'static [u32]>
where
    for<'a> T: LendingIterator<Item<'a> = &'a [u32]>,
{
    None::<<T as LendingIterator>::Item<'static>>
    // FIXME: Should not compile, but does, because we are trusting the where-clauses
    // and don't have implied bounds.
}

fn main() {}
