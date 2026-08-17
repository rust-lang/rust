//@ known-bug: #148621
type Foo<V> = impl std::fmt::Debug;

trait Identity<Q> {
    type T;
}

impl<Q> Clone for Foo<<() as Identity<Q>>::T> {
    fn clone(&self) -> Self {
        loop {}
    }
}

fn main() {}
