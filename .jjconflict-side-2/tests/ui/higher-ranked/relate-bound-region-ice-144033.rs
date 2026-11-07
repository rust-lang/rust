// Regression test for <https://github.com/rust-lang/rust/issues/144033>.
// This used to ICE with "cannot relate bound region" instead of emitting
// normal errors.

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
        //~^ ERROR expected `&I` to be an iterator that yields `&()`
        //~| ERROR mismatched types
    }
}

fn main() {}
