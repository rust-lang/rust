//! Regression test for https://github.com/rust-lang/rust/issues/13703

//@ check-pass

pub struct Foo<'a, 'b: 'a> { foo: &'a &'b isize }
pub fn foo<'a, 'b>(x: Foo<'a, 'b>, _o: Option<&   &   ()>) { let _y = x.foo; }
fn main() {}
