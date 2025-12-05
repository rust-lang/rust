// https://github.com/rust-lang/rust/issues/7673
//@ check-pass
#![allow(dead_code)]

/*

#7673 Polymorphically creating traits barely works

*/

pub fn main() {}

trait A {
    fn dummy(&self) { }
}

impl<T: 'static> A for T {}

fn owned2<T: 'static>(a: Box<T>) { a as Box<dyn A>; }
fn owned3<T: 'static>(a: Box<T>) { Box::new(a) as Box<dyn A>; }
