// revisions: no_drop_tracking drop_tracking drop_tracking_mir
// [drop_tracking] compile-flags: -Zdrop-tracking
// [drop_tracking_mir] compile-flags: -Zdrop-tracking-mir
#![feature(auto_traits)]
#![feature(negative_impls)]
// edition:2018

// This tests the unspecialized async-await-specific error when futures don't implement an
// auto trait (which is not Send or Sync) due to some type that was captured.

auto trait Qux {}

struct Foo;

impl !Qux for Foo {}

fn is_qux<T: Qux>(t: T) {}

async fn bar() {
    let x = Box::new(Foo);
    baz().await;
}

async fn baz() {}

fn main() {
    is_qux(bar());
    //~^ ERROR the trait bound `Foo: Qux` is not satisfied in `impl Future<Output = ()>`
}
