// check-pass
// edition:2018

use std::array::IntoIter;
use std::slice::Iter;

fn main() {
    let array = [0; 10];

    // Before 2021, the method dispatched to `IntoIterator for &[T; N]`,
    // which we continue to support for compatibility.
    let _: Iter<'_, i32> = array.into_iter();
    //~^ WARNING this method call currently resolves to `<&[T; N] as IntoIterator>::into_iter`
    //~| WARNING this was previously accepted by the compiler but is being phased out

    let _: Iter<'_, i32> = Box::new(array).into_iter();
    //~^ WARNING this method call currently resolves to `<&[T; N] as IntoIterator>::into_iter`
    //~| WARNING this was previously accepted by the compiler but is being phased out

    // But you can always use the trait method explicitly as an array.
    let _: IntoIter<i32, 10> = IntoIterator::into_iter(array);
}
