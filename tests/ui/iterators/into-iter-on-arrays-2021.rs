//@ check-pass
//@ edition:2021

use std::array::IntoIter;
use std::ops::Deref;
use std::rc::Rc;

fn main() {
    let array = [0; 10];

    // In 2021, the method dispatches to `IntoIterator for [T; N]`.
    let _: IntoIter<i32, 10> = array.into_iter();
    let _: IntoIter<i32, 10> = Box::new(array).into_iter();

    // The `array_into_iter` lint doesn't cover other wrappers that deref to an array.
    let _: IntoIter<i32, 10> = Rc::new(array).into_iter();
    let _: IntoIter<i32, 10> = Array(array).into_iter();

    // You can always use the trait method explicitly as an array.
    let _: IntoIter<i32, 10> = IntoIterator::into_iter(array);
}

/// User type that dereferences to an array.
struct Array([i32; 10]);

impl Deref for Array {
    type Target = [i32; 10];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
