//! Regression test for <https://github.com/rust-lang/rust/issues/20396>.
//@ check-pass

#![allow(dead_code)]

trait Foo<T> {
    fn noop(&self, _: T);
}

enum Bar<T> { Bla(T) }

struct Baz<'a> {
    inner: dyn for<'b> Foo<Bar<&'b ()>> + 'a,
}

fn main() {}
