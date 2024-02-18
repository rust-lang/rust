//@ check-pass
//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ edition:2021

#![feature(return_type_notation)]
//~^ WARN the feature `return_type_notation` is incomplete

trait Foo {
    async fn bar(&self);
}

struct Bar;
impl Foo for Bar {
    async fn bar(&self) {}
}

fn build<T>(_: T) where T: Foo<bar(): Send> {}

fn main() {
    build(Bar);
}
