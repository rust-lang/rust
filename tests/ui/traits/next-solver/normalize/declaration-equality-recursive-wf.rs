//@ check-pass
//@ revisions: current next assumptions
//@[current] compile-flags: -Znext-solver=no
//@[next] compile-flags: -Znext-solver=globally
//@[assumptions] compile-flags: -Znext-solver=globally -Zassumptions-on-binders

#![allow(dead_code)]

trait Outer {
    type Inner;

    fn object(&self) -> Object<Self::Inner>
    where
        Self::Inner: Inner<Outer = Self>;
}

trait Inner: Sized {
    type Outer: Outer<Inner = Self>;
}

struct Object<T: Inner>(std::marker::PhantomData<T>);
struct Storage<T: 'static>(T);
struct Wrapper<T: 'static>(T);

impl<T: 'static> Inner for Storage<T> {
    type Outer = Wrapper<T>;
}

impl<T: 'static> Outer for Wrapper<T> {
    type Inner = Storage<T>;

    fn object(&self) -> Object<Self::Inner> {
        Object(std::marker::PhantomData)
    }
}

fn main() {}
