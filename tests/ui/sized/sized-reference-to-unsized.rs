//! Check that a reference to a potentially unsized type (`&T`) is itself considered `Sized`.

//@ run-pass

#![allow(dead_code)]

fn bar<T: Sized>() {}
fn foo<T>() {
    bar::<&T>()
}
pub fn main() {}
