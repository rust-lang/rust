trait FooBar<'a> {
    type Item;
}

trait LendingIterator {
    type Item<'a>
    where
        Self: 'a;

    fn iter(&mut self) -> Option<Self::Item<'_>>;
}

fn main() {
    fn foo0(_: impl FooBar<_, Item = &u32>) {}
    //~^ the placeholder `_` is not allowed within types
    //~| trait takes 0 generic arguments but 1 generic
    fn foo1(_: impl LendingIterator<Item<'_> = &u32>) {}
    //~^ '_` cannot be used here [E0637]
}
