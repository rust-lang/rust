// https://github.com/rust-lang/rust/issues/55731
use std::marker::PhantomData;

trait DistributedIterator {
    fn reduce(self)
    where
        Self: Sized,
    {
        unreachable!()
    }
}

trait DistributedIteratorMulti<Source> {
    type Item;
}

struct Connect<I>(PhantomData<fn(I)>);
impl<I: for<'a> DistributedIteratorMulti<&'a ()>> DistributedIterator for Connect<I> where {}

struct Cloned<Source>(PhantomData<fn(Source)>);
impl<'a, Source> DistributedIteratorMulti<&'a Source> for Cloned<&'a Source> {
    type Item = ();
}

struct Map<I, F> {
    i: I,
    f: F,
}
impl<I: DistributedIteratorMulti<Source>, F, Source> DistributedIteratorMulti<Source> for Map<I, F>
where
    F: A<<I as DistributedIteratorMulti<Source>>::Item>,
{
    type Item = ();
}

trait A<B> {}

struct X;
impl A<()> for X {}

fn multi<I>(_reducer: I)
where
    I: for<'a> DistributedIteratorMulti<&'a ()>,
{
    DistributedIterator::reduce(Connect::<I>(PhantomData))
}

fn main() {
    multi(Map { //~ ERROR implementation of `DistributedIteratorMulti` is not general enough
        i: Cloned(PhantomData),
        f: X,
    });
}
