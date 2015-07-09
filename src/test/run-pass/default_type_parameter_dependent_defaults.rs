use std::marker::PhantomData;

struct Foo<T,U=T> { data: PhantomData<(T, U)> }

fn main() {
    let foo = Foo { data: PhantomData };
}
