//@ compile-flags: -Znext-solver

trait Foo {
    type Assoc;
}

trait Bar {}
fn needs_bar<S: Bar>() {}

fn test<T: Foo<Assoc = <T as Foo>::Assoc>>() {
    needs_bar::<T::Assoc>();
    //~^ ERROR overflow evaluating the requirement `<T as Foo>::Assoc == _`
    //~| ERROR overflow evaluating the requirement `<T as Foo>::Assoc == _`
    //~| ERROR overflow evaluating the requirement `<T as Foo>::Assoc == _`
    //~| ERROR overflow evaluating the requirement `<T as Foo>::Assoc == _`
    //~| ERROR overflow evaluating the requirement `<T as Foo>::Assoc: Sized`
    //~| ERROR overflow evaluating the requirement `<T as Foo>::Assoc: Bar`
}

fn main() {}
