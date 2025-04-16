//@ compile-flags: -Znext-solver
//@ edition: 2021
//@ revisions: pass fail
//@[pass] check-pass

use std::future::Future;

fn needs_async(_: impl Future<Output = i32>) {}

#[cfg(fail)]
fn main() {
    needs_async(async {});
    //[fail]~^ ERROR type mismatch resolving `() == i32`
}

#[cfg(pass)]
fn main() {
    needs_async(async { 1i32 });
}
