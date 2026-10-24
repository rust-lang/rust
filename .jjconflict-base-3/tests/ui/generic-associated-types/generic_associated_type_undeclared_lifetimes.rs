use std::ops::Deref;

trait Iterable {
    type Item<'a>;
    type Iter<'a>: Iterator<Item = Self::Item<'a>>
        + Deref<Target = Self::Item<'b>>;
    //~^ ERROR undeclared lifetime

    fn iter<'a>(&'a self) -> Self::Iter<'undeclared>;
    //~^ ERROR undeclared lifetime
}

fn main() {}
