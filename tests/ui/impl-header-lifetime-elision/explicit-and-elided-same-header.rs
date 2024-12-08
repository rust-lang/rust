//@ run-pass

#![allow(warnings)]

// This works for functions...
fn foo<'a>(x: &str, y: &'a str) {}

// ...so this should work for impls
impl<'a> Foo<&str> for &'a str {}
trait Foo<T> {}

fn main() {
}
