// build-pass
#![allow(dead_code)]
trait Foo {
    fn foo(&self);
}

fn foo<'a>(s: &'a mut ()) where &'a mut (): Foo {
    s.foo();
}
fn main() {}
