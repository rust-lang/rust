// build-pass (FIXME(62277): could be check-pass?)
#![allow(dead_code)]
// pretty-expanded FIXME #23616

/*

#7673 Polymorphically creating traits barely works

*/

#![feature(box_syntax)]

pub fn main() {}

trait A {
    fn dummy(&self) { }
}

impl<T: 'static> A for T {}

fn owned2<T: 'static>(a: Box<T>) { a as Box<dyn A>; }
fn owned3<T: 'static>(a: Box<T>) { box a as Box<dyn A>; }
