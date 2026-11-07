//@ check-pass

#![feature(return_type_notation)]

trait Trait {
    fn method() -> impl Sized;
}

fn is_send(_: impl Send) {}

struct W<T>(T);

impl<T> W<T> {
    fn test()
    where
        T: Trait,
        T::method(..): Send,
    {
        is_send(T::method());
    }
}

fn main() {}
