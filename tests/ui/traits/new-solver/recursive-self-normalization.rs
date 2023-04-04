// compile-flags: -Ztrait-solver=next

trait Foo {
    type Assoc;
}

trait Bar {}
fn needs_bar<S: Bar>() {}

fn test<T: Foo<Assoc = <T as Foo>::Assoc>>() {
    needs_bar::<T::Assoc>();
    //~^ ERROR type annotations needed
}

fn main() {}
