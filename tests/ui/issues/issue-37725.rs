//@ build-pass
// compiler-opts: -Zmir-opt-level=2

#![allow(dead_code)]
trait Foo {
    fn foo(&self);
}

fn foo<'a>(s: &'a mut ()) where &'a mut (): Foo {
    s.foo();
}
fn main() {}
