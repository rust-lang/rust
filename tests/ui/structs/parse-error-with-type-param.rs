//@ run-rustfix
// #141403
#![allow(dead_code)]

#[derive(Clone)]
struct B<T> {
    a: A<(T, u32)> // <- note, comma is missing here
    /// asdf
    //~^ ERROR found a documentation comment that doesn't document anything
    b: u32,
}
#[derive(Clone)]
struct A<T>(T);
fn main() {}
