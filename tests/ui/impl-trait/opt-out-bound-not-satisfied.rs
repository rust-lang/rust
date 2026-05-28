//! Regression test for ICE https://github.com/rust-lang/rust/issues/135730
//! This used

use std::future::Future;
fn foo() -> impl ?Future<Output = impl Send> {
    //~^ ERROR: bound modifier `?` can only be applied to `Sized`
    //~| ERROR: bound modifier `?` can only be applied to `Sized`
    ()
}

pub fn main() {}
