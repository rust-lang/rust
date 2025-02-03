//@ check-pass

// Make sure that we still deduce outlives bounds from supertrait projections
// and require them for well-formedness.

trait Trait {
    type Assoc;
}

trait Bar {
    type Assoc;
}

trait Foo<'a, T: 'a>: Bar<Assoc = &'a T> {

}

fn outlives<'a, T: 'a>() {}

fn implied_outlives<'a, T: Trait>(x: &dyn Foo<'a, T::Assoc>) {
    outlives::<'a, T::Assoc>();
}

fn main() {}
