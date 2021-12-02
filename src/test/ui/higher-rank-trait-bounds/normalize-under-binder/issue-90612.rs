// check-pass

#![feature(generic_associated_types)]

use std::marker::PhantomData;

trait Family: Sized {
    type Item<'a>;

    fn apply_all<F>(&self, f: F)
    where
        F: FamilyItemFn<Self> { }
}

struct Array<T>(PhantomData<T>);

impl<T: 'static> Family for Array<T> {
    type Item<'a> = &'a T;
}

trait FamilyItemFn<T: Family> {
    fn apply(&self, item: T::Item<'_>);
}

impl<T, F> FamilyItemFn<T> for F
where
    T: Family,
    for<'a> F: Fn(T::Item<'a>)
{
    fn apply(&self, item: T::Item<'_>) {
        (*self)(item);
    }
}

fn process<T: 'static>(array: Array<T>) {
    // Works
    array.apply_all(|x: &T| { });

    // ICE: NoSolution
    array.apply_all(|x: <Array<T> as Family>::Item<'_>| { });
}

fn main() {}
