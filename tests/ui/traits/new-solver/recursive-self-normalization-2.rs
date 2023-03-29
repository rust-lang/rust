// compile-flags: -Ztrait-solver=next

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
    //~^ ERROR type annotations needed
}

fn main() {}
