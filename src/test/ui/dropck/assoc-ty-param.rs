struct Foo<T>
where
    T: Iterator,
    T::Item: Send,
{
    t: T,
}

impl<T, I> Drop for Foo<T>
//~^ ERROR `Drop` impl requires `I: Sized`
where
    T: Iterator<Item = I>,
    //~^ ERROR `Drop` impl requires `<T as Iterator>::Item == I`
    I: Send,
    //~^ ERROR `Drop` impl requires `I: Send`

{
    fn drop(&mut self) { }
}

fn main() { }