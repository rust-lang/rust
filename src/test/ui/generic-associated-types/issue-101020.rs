#![feature(generic_associated_types)]

pub trait LendingIterator {
    type Item<'a>
    where
        Self: 'a;

    fn consume<F>(self, _f: F)
    where
        Self: Sized,
        for<'a> Self::Item<'a>: FuncInput<'a, Self::Item<'a>>,
    {
    }
}

impl<I: LendingIterator + ?Sized> LendingIterator for &mut I {
    type Item<'a> = I::Item<'a> where Self: 'a;
}
struct EmptyIter;
impl LendingIterator for EmptyIter {
    type Item<'a> = &'a mut () where Self:'a;
}
pub trait FuncInput<'a, F>
where
    F: Foo<Self>,
    Self: Sized,
{
}
impl<'a, T, F: 'a> FuncInput<'a, F> for T where F: Foo<T> {}
trait Foo<T> {}

fn map_test() {
    (&mut EmptyIter).consume(());
    //~^ ERROR the trait bound `for<'a> &'a mut (): Foo<&'a mut ()>` is not satisfied
}

fn main() {}
