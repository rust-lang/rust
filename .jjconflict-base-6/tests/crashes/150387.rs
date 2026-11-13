//@ known-bug: #150387
#![feature(min_specialization)]
#![allow(dead_code)]

struct Thing<T>(T) where [T]: Sized;

impl<T> Drop for Thing<T> where [T]: Sized {
    default fn drop(&mut self) {}
}
impl<T> Drop for Thing<T> where [T]: Sized {
    fn drop(&mut self) {}
}
fn main() {}
