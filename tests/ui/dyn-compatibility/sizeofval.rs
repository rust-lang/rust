//@ run-pass
//! This test and `sized-*.rs` and `pointeesized.rs` test that dyn-compatibility correctly
//! handles sizedness traits, which are special in several parts of the compiler.
#![feature(sized_hierarchy)]
use std::marker::SizeOfVal;

trait Foo: std::fmt::Debug + SizeOfVal {}

impl<T: std::fmt::Debug + SizeOfVal> Foo for T {}

fn unsize_sized<T: 'static>(x: Box<T>) -> Box<dyn SizeOfVal> {
    x
}

fn unsize_subtrait(x: Box<dyn Foo>) -> Box<dyn SizeOfVal> {
    x
}

fn main() {
    let _bx = unsize_sized(Box::new(vec![1, 2, 3]));

    let bx: Box<dyn Foo> = Box::new(vec![1, 2, 3]);
    let _ = format!("{bx:?}");
    let _bx = unsize_subtrait(bx);
}
