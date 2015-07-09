use std::marker::PhantomData;

trait Id {
    type This;
}

impl<A> Id for A {
    type This = A;
}

struct Foo<X: Default = usize, Y = <X as Id>::This> {
    data: PhantomData<(X, Y)>
}

impl<X: Default, Y> Foo<X, Y> {
    fn new() -> Foo<X, Y> {
        Foo { data: PhantomData }
    }
}

fn main() {
    let foo = Foo::new();
}
