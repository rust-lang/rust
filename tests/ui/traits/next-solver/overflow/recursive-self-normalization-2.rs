//@ compile-flags: -Znext-solver

trait Foo1 {
    type Assoc1;
}

trait Foo2 {
    type Assoc2;
}

trait Bar {}
fn needs_bar<S: Bar>() {}

fn test<T: Foo1<Assoc1 = <T as Foo2>::Assoc2> + Foo2<Assoc2 = <T as Foo1>::Assoc1>>() {
    needs_bar::<T::Assoc1>();
    //~^ ERROR: the trait bound `<T as Foo2>::Assoc2: Bar` is not satisfied
    //~| ERROR: the size for values of type `<T as Foo2>::Assoc2` cannot be known at compilation time
}

fn main() {}
