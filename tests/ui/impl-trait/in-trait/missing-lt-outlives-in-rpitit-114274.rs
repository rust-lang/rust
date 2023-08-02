#![feature(return_position_impl_trait_in_trait)]

trait Iterable {
    type Item<'a>
    where
        Self: 'a;

    fn iter(&self) -> impl Iterator<Item = Self::Item<'missing>>;
    //~^ ERROR use of undeclared lifetime name `'missing`
}

fn main() {}
