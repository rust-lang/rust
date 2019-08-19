// build-pass (FIXME(62277): could be check-pass?)
// pretty-expanded FIXME #23616

#![allow(dead_code)]

trait Foo<T> {
    fn noop(&self, _: T);
}

enum Bar<T> { Bla(T) }

struct Baz<'a> {
    inner: dyn for<'b> Foo<Bar<&'b ()>> + 'a,
}

fn main() {}
