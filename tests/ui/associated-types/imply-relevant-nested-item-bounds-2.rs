//@ check-pass
//@ revisions: current next
//@[next] compile-flags: -Znext-solver

trait Trait
where
    Self::Assoc: Clone,
{
    type Assoc;
}

fn foo<T: Trait>(x: &T::Assoc) -> T::Assoc {
    x.clone()
}

trait Trait2
where
    Self::Assoc: Iterator,
    <Self::Assoc as Iterator>::Item: Clone,
{
    type Assoc;
}

fn foo2<T: Trait2>(x: &<T::Assoc as Iterator>::Item) -> <T::Assoc as Iterator>::Item {
    x.clone()
}

fn main() {}
