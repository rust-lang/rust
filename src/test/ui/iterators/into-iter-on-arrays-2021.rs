// check-pass
// edition:2021
// compile-flags: -Zunstable-options

use std::array::IntoIter;

fn main() {
    let array = [0; 10];

    // In 2021, the method dispatches to `IntoIterator for [T; N]`.
    let _: IntoIter<i32, 10> = array.into_iter();
    let _: IntoIter<i32, 10> = Box::new(array).into_iter();

    // And you can always use the trait method explicitly as an array.
    let _: IntoIter<i32, 10> = IntoIterator::into_iter(array);
}
