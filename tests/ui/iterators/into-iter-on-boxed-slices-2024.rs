//@ check-pass
//@ edition:2024

use std::ops::Deref;
use std::rc::Rc;
use std::vec::IntoIter;

fn main() {
    let boxed_slice = vec![0; 10].into_boxed_slice();

    // In 2021, the method dispatches to `IntoIterator for [T; N]`.
    let _: IntoIter<i32> = boxed_slice.clone().into_iter();

    // And through other boxes.
    let _: IntoIter<i32> = Box::new(boxed_slice.clone()).into_iter();

    // You can always use the trait method explicitly as a boxed_slice.
    let _: IntoIter<i32> = IntoIterator::into_iter(boxed_slice.clone());
}
