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
