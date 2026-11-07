//@ known-bug: #140381
pub trait Foo<T> {}
pub trait Lend {
    type From<'a>
    where
        Self: 'a;
    fn lend(from: Self::From<'_>) -> impl Foo<Self::From<'_>>;
}

impl<T, F> Lend for (T, F) {
    type From<'a> = ();

    fn lend(from: Self::From<'_>) -> impl Foo<Self::From<'_>> {
        from
    }
}
