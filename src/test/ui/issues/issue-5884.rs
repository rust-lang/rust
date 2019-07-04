// build-pass (FIXME(62277): could be check-pass?)
#![allow(dead_code)]
// pretty-expanded FIXME #23616

#![feature(box_syntax)]

pub struct Foo {
    a: isize,
}

struct Bar<'a> {
    a: Box<Option<isize>>,
    b: &'a Foo,
}

fn check(a: Box<Foo>) {
    let _ic = Bar{ b: &*a, a: box None };
}

pub fn main(){}
