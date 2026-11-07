//@ compile-flags: -Znext-solver
//@ check-pass

// Regression test for #119607.

pub trait IntoFoo {
    type Item;
    type IntoIter: Foo<Item = Self::Item>;

    fn into_iter(self) -> Self::IntoIter;
}

pub trait Foo {
    type Item;

    fn next(self) -> Option<Self::Item>;
}

pub fn foo<'a, Iter1, Elem1>(a: &'a Iter1)
where
    &'a Iter1: IntoFoo<Item = Elem1>,
{
    a.into_iter().next();
}

fn main() {}
