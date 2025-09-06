#![allow(incomplete_features)]
#![feature(field_projections)]

use std::field::field_of;

struct MyStruct(());

impl Clone for field_of!(MyStruct, 0) {
    fn clone(&self) -> Self {
        *self
    }
}

impl Copy for field_of!(MyStruct, 0) {}

fn assert_copy<T: Copy>() {}

fn main() {
    assert_copy::<field_of!(MyStruct, 0)>();
}
