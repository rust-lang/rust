//@ compile-flags: -Znext-solver

trait Foo {
    type Assoc;
}

trait Bar {}
fn needs_bar<S: Bar>() {}

fn test<T: Foo<Assoc = <T as Foo>::Assoc>>() {
    needs_bar::<T::Assoc>();
    //~^ ERROR the trait bound `<T as Foo>::Assoc: Bar` is not satisfied
}

fn main() {}
