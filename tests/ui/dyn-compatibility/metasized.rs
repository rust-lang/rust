//@ run-pass
//! This test and `sized-*.rs` and `pointeesized.rs` test that dyn-compatibility correctly
//! handles sizedness traits, which are special in several parts of the compiler.
#![feature(sized_hierarchy)]
use std::marker::MetaSized;

trait Foo: std::fmt::Debug + MetaSized {}

impl<T: std::fmt::Debug + MetaSized> Foo for T {}

fn unsize_sized<T: 'static>(x: Box<T>) -> Box<dyn MetaSized> {
    x
}

fn unsize_subtrait(x: Box<dyn Foo>) -> Box<dyn MetaSized> {
    x
}

fn main() {
    let _bx = unsize_sized(Box::new(vec![1, 2, 3]));

    let bx: Box<dyn Foo> = Box::new(vec![1, 2, 3]);
    let _ = format!("{bx:?}");
    let _bx = unsize_subtrait(bx);
}
