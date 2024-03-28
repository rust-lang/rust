// check-pass
// edition:2024

use std::ops::Deref;
use std::rc::Rc;
use std::vec::IntoIter;

fn main() {
    let boxed_slice = vec![0; 10].into_boxed_slice();

    // In 2021, the method dispatches to `IntoIterator for [T; N]`.
    let _: IntoIter<i32> = boxed_slice.into_iter();

    // The `boxed_slice_into_iter` lint doesn't cover other wrappers that deref to a boxed_slice.
    let _: IntoIter<i32> = Rc::new(boxed_slice).into_iter();
    let _: IntoIter<i32> = Array(boxed_slice).into_iter();

    // You can always use the trait method explicitly as a boxed_slice.
    let _: IntoIter<i32> = IntoIterator::into_iter(boxed_slice);
}

/// User type that dereferences to a boxed slice.
struct Array(Box<i32>);

impl Deref for Array {
    type Target = Box<i32>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
