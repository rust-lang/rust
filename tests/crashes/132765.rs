//@ known-bug: #132765

trait LendingIterator {
    type Item<'q>;
    fn for_each(&self, _f: Box<fn(Self::Item<'_>)>) {}
}

fn f(_: ()) {}

fn main() {
    LendingIterator::for_each(&(), f);
}
