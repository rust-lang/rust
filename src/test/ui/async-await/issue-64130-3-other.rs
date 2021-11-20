#![feature(auto_traits)]
#![feature(negative_impls)]
// edition:2018

// This tests the the unspecialized async-await-specific error when futures don't implement an
// auto trait (which is not Send or Sync) due to some type that was captured.

auto trait Qux {}

struct Foo;

impl !Qux for Foo {}

fn is_qux<T: Qux>(t: T) {}

async fn bar() {
    let x = Foo;
    baz().await;
}

async fn baz() {}

fn main() {
    is_qux(bar());
    //~^ ERROR the trait bound `Foo: Qux` is not satisfied in `impl Future<Output = ()>`
}
