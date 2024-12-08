use std::marker;

struct Foo<A, B, C = (A, B)>(
    marker::PhantomData<(A,B,C)>);

impl<A, B, C> Foo<A, B, C> {
    fn new() -> Foo<A, B, C> {Foo(marker::PhantomData)}
}

fn main() {
    Foo::<isize>::new();
    //~^ ERROR struct takes at least 2 generic arguments but 1 generic argument
}
