// run-pass
#![allow(unused_variables)]
/*
# ICE when returning struct with reference to trait

A function which takes a reference to a trait and returns a
struct with that reference results in an ICE.

This does not occur with concrete types, only with references
to traits.
*/


// original
trait Inner {
    fn print(&self);
}

impl Inner for isize {
    fn print(&self) { print!("Inner: {}\n", *self); }
}

struct Outer<'a> {
    inner: &'a (Inner+'a)
}

impl<'a> Outer<'a> {
    fn new(inner: &Inner) -> Outer {
        Outer {
            inner: inner
        }
    }
}

pub fn main() {
    let inner: isize = 5;
    let outer = Outer::new(&inner as &Inner);
    outer.inner.print();
}


// minimal
pub trait MyTrait<T> {
    fn dummy(&self, t: T) -> T { panic!() }
}

pub struct MyContainer<'a, T:'a> {
    foos: Vec<&'a (MyTrait<T>+'a)> ,
}

impl<'a, T> MyContainer<'a, T> {
    pub fn add (&mut self, foo: &'a MyTrait<T>) {
        self.foos.push(foo);
    }
}
