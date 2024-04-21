// Incremental stabilization of `non-generic associated types` for non-generic associated types.

mod stabilized {
    trait FooBar<'a> {
        type Item;
    }

    fn foo0(x: impl Iterator<Item = &u32>) {
    }

    fn foo1<'b>(_: impl FooBar<'b, Item = &u32>) {
    }
}

mod not_stabilized {
    trait FooBar<'a> {
        type Item;
    }

    trait LendingIterator {
        type Item<'a>
        where
            Self: 'a;
    }

    fn foo0(_: impl LendingIterator<Item<'_> = &u32>) {}
    //~^ ERROR `'_` cannot be used here
    //~| ERROR anonymous lifetimes in `impl Trait` are unstable

    fn foo1(_: impl FooBar<'_, Item = &u32>) {}
}

fn main() {}
