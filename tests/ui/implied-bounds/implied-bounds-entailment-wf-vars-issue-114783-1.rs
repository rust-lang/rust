//@ check-pass

pub trait Foo {
    type Error: Error;

    fn foo(&self, stream: &<Self::Error as Error>::Span);
}

pub struct Wrapper<Inner>(Inner);

impl<E: Error, Inner> Foo for Wrapper<Inner>
where
    Inner: Foo<Error = E>,
{
    type Error = E;

    fn foo(&self, stream: &<Self::Error as Error>::Span) {
        todo!()
    }
}

pub trait Error {
    type Span;
}

fn main() {}
