pub trait Iterable {
    type Item<'a>
    where
        Self: 'a;

    fn iter(&self) -> impl Iterator;
}

impl<'a, I: 'a + Iterable> Iterable for &'a I {
    type Item = u32;
    //~^ ERROR lifetime parameters or bounds on associated type `Item` do not match the trait declaration

    fn iter(&self) -> impl for<'missing> Iterator<Item = Self::Item<'missing>> {}
    //~^ ERROR binding for associated type `Item` references lifetime `'missing`
    //~| ERROR binding for associated type `Item` references lifetime `'missing`
    //~| ERROR `()` is not an iterator
}

fn main() {}
