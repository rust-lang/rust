//@ check-pass

#![feature(return_type_notation)]

trait Trait {
    fn method(&self) -> impl Sized;
}

impl Trait for () {
    fn method(&self) -> impl Sized {}
}

struct Struct<T>(T);

// This test used to fail a debug assertion since we weren't resolving the item
// for `T::method(..)` correctly, leading to two bound vars being given the
// index 0. The solution is to look at both generics of `test` and its parent impl.

impl<T> Struct<T>
where
    T: Trait,
{
    fn test()
    where
        T::method(..): Send
    {}
}

fn main() {
    Struct::<()>::test();
}
