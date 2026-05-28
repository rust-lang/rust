//@ check-pass
//! This test checks that dropck knows that ManuallyDrop does not drop its field.
#![feature(dropck_eyepatch)]

use std::mem::ManuallyDrop;

struct S<T>(ManuallyDrop<T>);

unsafe impl<#[may_dangle] T> Drop for S<T> {
    fn drop(&mut self) {}
}

struct NonTrivialDrop<'a>(&'a str);
impl<'a> Drop for NonTrivialDrop<'a> {
    fn drop(&mut self) {}
}

fn main() {
    let s = String::from("string");
    let _t = S(ManuallyDrop::new(NonTrivialDrop(&s)));
    drop(s);
}
