// https://github.com/rust-lang/rust/issues/7268
//@ check-pass
#![allow(dead_code)]

fn foo<T: 'static>(_: T) {}

fn bar<T>(x: &'static T) {
    foo(x);
}
fn main() {}
