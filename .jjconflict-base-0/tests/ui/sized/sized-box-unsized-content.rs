//! Check that `Box<T>` is `Sized`, even when `T` is a dynamically sized type.

//@ run-pass

#![allow(dead_code)]

fn bar<T: Sized>() {}
fn foo<T>() {
    bar::<Box<T>>()
}
pub fn main() {}
