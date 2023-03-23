// Foo<Vec<X>> shouldn't imply X: 'static.
// We don't use region constraints from trait impls in implied bounds.

trait Trait {
    type Assoc;
}

impl<X: 'static> Trait for Vec<X> {
    type Assoc = ();
}

struct Foo<T: Trait>(T)
where
    T::Assoc: 'static, // any predicate naming T::Assoc
;

fn foo<X>(_: Foo<Vec<X>>) {}
//~^ ERROR `X` may not live long enough
//~| ERROR `X` may not live long enough

fn main() {}
