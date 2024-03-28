// Regression test for #122098. Has been slightly minimized, fixed by #122189.
trait LendingIterator: Sized {
    type Item<'q>;

    fn for_each(self, f: Box<dyn FnMut(Self::Item<'_>)>) {}
}

struct Query;
fn main() {
    LendingIterator::for_each(Query, Box::new);
    //~^ ERROR the trait bound `Query: LendingIterator` is not satisfied
    //~| ERROR mismatched types
    //~| ERROR the trait bound `Query: LendingIterator` is not satisfied
}
