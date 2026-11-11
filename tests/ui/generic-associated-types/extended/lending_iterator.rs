//@ known-bug: #133805

pub trait FromLendingIterator<A>: Sized {
    fn from_iter<T: for<'x> LendingIterator<Item<'x> = A>>(iter: T) -> Self;
}

impl<A> FromLendingIterator<A> for Vec<A> {
    fn from_iter<I: for<'x> LendingIterator<Item<'x> = A>>(mut iter: I) -> Self {
        let mut v = vec![];
        while let Some(item) = iter.next() {
            v.push(item);
        }
        v
    }
}

pub trait LendingIterator {
    type Item<'z>
    where
        Self: 'z;
    fn next(&mut self) -> Option<Self::Item<'_>>;

    fn collect<A, B: FromLendingIterator<A>>(self) -> B
    where
        Self: Sized,
        Self: for<'q> LendingIterator<Item<'q> = A>,
    {
        <B as FromLendingIterator<A>>::from_iter(self)
    }
}

fn main() {}
