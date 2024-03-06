//@ build-pass
#![allow(dead_code)]
//@ pretty-expanded FIXME #23616

pub struct Foo {
    a: isize,
}

struct Bar<'a> {
    a: Box<Option<isize>>,
    b: &'a Foo,
}

fn check(a: Box<Foo>) {
    let _ic = Bar{ b: &*a, a: Box::new(None) };
}

pub fn main(){}
