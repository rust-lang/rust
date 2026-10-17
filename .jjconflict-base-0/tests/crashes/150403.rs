//@ known-bug: #150403
#![feature(non_lifetime_binders)]

trait A {
    type GAT<T>: A;
    fn foo<T>(self, t: T) -> Self::GAT<T>
    where
        Self: Sized;
}

trait B: A where
    for<T> Self::GAT<T>: B,
{
    fn bar<T>(self) -> Self::GAT<T>
    where
        Self: Sized;

    fn baz<T>(self, t: T) -> Self::GAT<T>
    where
        Self: Sized,
    {
        self.foo(t).bar()
    }
}

fn main() {}
