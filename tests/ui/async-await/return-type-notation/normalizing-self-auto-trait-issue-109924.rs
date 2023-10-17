// revisions: current next
//[current] known-bug: #109924
//[next] check-pass
//[next] compile-flags: -Ztrait-solver=next
// edition:2021

#![feature(return_type_notation)]
//[next]~^ WARN the feature `return_type_notation` is incomplete

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
