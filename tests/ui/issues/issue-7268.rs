//@ check-pass
#![allow(dead_code)]

fn foo<T: 'static>(_: T) {}

fn bar<T>(x: &'static T) {
    foo(x);
}
fn main() {}
