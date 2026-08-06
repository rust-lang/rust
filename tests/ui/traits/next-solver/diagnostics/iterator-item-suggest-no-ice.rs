//! Regression test for <https://github.com/rust-lang/rust/issues/158317>
//@ compile-flags: -Znext-solver

trait FooMut {
    fn bar<I>(&self, _: I)
    where
        for<'b> &'b I: Iterator<Item = &'b ()>;

    fn bar<I>(&self, _: I)
    //~^ ERROR: the name `bar` is defined multiple times
    where
        I: Iterator,
    {
        let collection = vec![_I].iter().map(|x| ());
        //~^ ERROR: cannot find value `_I` in this scope
        self.bar(collection);
        //~^ ERROR: `&'b _` is not an iterator
        //~| ERROR: type mismatch resolving `<&_ as Iterator>::Item == &()`
    }
}

fn main() {}
