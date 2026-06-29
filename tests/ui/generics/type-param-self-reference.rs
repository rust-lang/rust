//! Regression test for <https://github.com/rust-lang/rust/issues/18183>.

pub struct Foo<Bar=Bar>(Bar); //~ ERROR E0128
pub struct Baz(Foo);
fn main() {}
