trait LendingIterator {
    type Item<'a>
    where
        Self: 'a;
}

impl LendingIterator for &str {
    type Item<'a> = () where Self:'a;
}

fn trivial_bound<I>(_: I)
where
    I: LendingIterator,
    for<'a> I::Item<'a>: Sized,
{
}

fn fails(iter: &str) {
    trivial_bound(iter);
    //~^ ERROR borrowed data escapes
}

fn main() {}
