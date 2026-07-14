// Regression test for <https://github.com/rust-lang/rust/issues/144033>.
// A delegating impl whose method drops the trait's HRTB where-clause used to
// ICE with "cannot relate bound region" instead of emitting normal errors.

trait FooMut {
    type Baz: 'static;
    fn bar<'a, I>(self, iterator: &'a I)
    where
        for<'b> &'b I: IntoIterator<Item = &'b &'a Self::Baz>;
}
struct DelegatingFooMut<T> {}
//~^ ERROR type parameter `T` is never used

impl<T> FooMut for DelegatingFooMut<T>
where
    T: FooMut,
{
    type Baz = DelegatingBaz<T::Baz>;
    fn bar<'a, I>(self, collection: &'a I)
    //~^ ERROR lifetime parameters do not match the trait definition
    where
        for<'b> &'b I: IntoIterator,
    {
        let collection = collection.into_iter().map(|b| &b);
        self.bar(collection)
        //~^ ERROR type mismatch resolving
        //~| ERROR mismatched types
    }
}
struct DelegatingBaz<T>;
//~^ ERROR type parameter `T` is never used

fn main() {}
