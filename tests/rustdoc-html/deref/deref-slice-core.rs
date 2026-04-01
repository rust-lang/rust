// https://github.com/rust-lang/rust/issues/95325
//
// Show methods reachable from Deref of primitive.
#![no_std]

use core::ops::Deref;

//@ has 'deref_slice_core/struct.MyArray.html'
//@ has '-' '//*[@id="deref-methods-%5BT%5D"]' 'Methods from Deref<Target = [T]>'
//@ has '-' '//*[@class="impl-items"]//*[@id="method.len"]' 'pub fn len(&self)'

pub struct MyArray<T> {
    array: [T; 10],
}

impl<T> Deref for MyArray<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        &self.array
    }
}
