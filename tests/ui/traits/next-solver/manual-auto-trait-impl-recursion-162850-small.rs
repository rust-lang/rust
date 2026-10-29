//@ check-pass
//@ compile-flags: -Znext-solver=globally

// Keep the `RegionEq` direction consistent when registering and collecting
// region constraints to avoid the overflow in #162850

#![allow(unused)]

struct RawVec<T>(std::marker::PhantomData<T>);

unsafe impl<T: Send> Send for RawVec<T> {}

pub struct MyVec<T>(RawVec<T>);

impl<T> MyVec<T> {
    pub fn new() -> Self {
        panic!()
    }
}

pub trait IntoParallelIterator {
    type Iter;

    fn into_par_iter(self) -> Self::Iter;
}

impl<T: Send> IntoParallelIterator for MyVec<T> {
    type Iter = T;

    fn into_par_iter(self) -> Self::Iter {
        todo!()
    }
}

struct Recursive<R>(MyVec<Recursive<R>>, R);

trait HasAssoc<'a> {
    type Assoc;
}

fn foo<'a, T>()
where
    T: HasAssoc<'a>,
    T::Assoc: Send,
{
    _ = MyVec::<Recursive<T::Assoc>>::new().into_par_iter();
}

fn main() {}
