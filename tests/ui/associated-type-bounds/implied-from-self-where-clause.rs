// Make sure that, like associated type where clauses on traits, we gather item
// bounds for RPITITs from RTN where clauses.

//@ check-pass

#![feature(return_type_notation)]

trait Foo
where
    Self::method(..): Send,
{
    fn method() -> impl Sized;
}

fn is_send(_: impl Send) {}

fn test<T: Foo>() {
    is_send(T::method());
}

fn main() {}
