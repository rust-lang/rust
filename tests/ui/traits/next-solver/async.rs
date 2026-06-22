//@ compile-flags: -Znext-solver
//@ edition: 2021
//@ revisions: pass fail
//@[pass] check-pass

use std::future::Future;

fn needs_async(_: impl Future<Output = i32>) {}

#[cfg(fail)]
fn main() {
    needs_async(async {});
    //[fail]~^ ERROR expected `{async block@$DIR/async.rs:12:17: 12:22}` to be a future that resolves to `i32`, but it resolves to `()`
}

#[cfg(pass)]
fn main() {
    needs_async(async { 1i32 });
}
