// build-pass (FIXME(62277): could be check-pass?)
#![allow(dead_code)]
// Regression test for #17746

fn main() {}

struct A;

impl A {
    fn b(&mut self) {
        self.a()
    }
}

trait Foo {
    fn dummy(&self) {}
}
trait Bar {
    fn a(&self);
}

impl Foo for A {}
impl<T> Bar for T where T: Foo {
    fn a(&self) {}
}
