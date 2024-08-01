//@ check-pass
//@ edition:2018

use std::ops::Deref;
use std::rc::Rc;
use std::slice::Iter;
use std::vec::IntoIter;

fn main() {
    let boxed_slice = vec![0; 10].into_boxed_slice();

    // Before 2024, the method dispatched to `IntoIterator for Box<[T]>`,
    // which we continue to support for compatibility.
    let _: Iter<'_, i32> = boxed_slice.into_iter();
    //~^ WARNING this method call resolves to `<&Box<[T]> as IntoIterator>::into_iter`
    //~| WARNING this changes meaning

    let _: Iter<'_, i32> = Box::new(boxed_slice.clone()).into_iter();
    //~^ WARNING this method call resolves to `<&Box<[T]> as IntoIterator>::into_iter`
    //~| WARNING this changes meaning

    let _: Iter<'_, i32> = Rc::new(boxed_slice.clone()).into_iter();
    //~^ WARNING this method call resolves to `<&Box<[T]> as IntoIterator>::into_iter`
    //~| WARNING this changes meaning
    let _: Iter<'_, i32> = Array(boxed_slice.clone()).into_iter();
    //~^ WARNING this method call resolves to `<&Box<[T]> as IntoIterator>::into_iter`
    //~| WARNING this changes meaning

    // But you can always use the trait method explicitly as an boxed_slice.
    let _: IntoIter<i32> = IntoIterator::into_iter(boxed_slice);

    for _ in (Box::new([1, 2, 3]) as Box<[_]>).into_iter() {}
    //~^ WARNING this method call resolves to `<&Box<[T]> as IntoIterator>::into_iter`
    //~| WARNING this changes meaning
}

/// User type that dereferences to a boxed slice.
struct Array(Box<[i32]>);

impl Deref for Array {
    type Target = Box<[i32]>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
