#![feature(generic_const_items)]
#![allow(incomplete_features, dead_code)]

//@ check-pass

trait Foo<T> {
    const BAR: bool
    where
        Self: Sized;
}

trait Cake {}
impl Cake for () {}

fn foo(_: &dyn Foo<()>) {}
fn bar(_: &dyn Foo<i32>) {}

fn main() {}
