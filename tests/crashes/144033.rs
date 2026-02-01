//@ known-bug: rust-lang/rust#144033
trait FooMut {
    fn bar<I>(self, _: I)
    where
        for<'b> &'b I: Iterator<Item = &'b ()>;
}

impl FooMut for () {
    fn bar<I>(self, _: I)
    where
        for<'b> &'b I: Iterator,
    {
        let collection = std::iter::empty::<()>().map(|_| &());
        self.bar(collection)
    }
}
