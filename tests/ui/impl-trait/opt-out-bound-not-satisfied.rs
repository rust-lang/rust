//! Regression test for ICE https://github.com/rust-lang/rust/issues/135730
//! This used

use std::future::Future;
fn foo() -> impl ?Future<Output = impl Send> {
    //~^ ERROR: relaxing a default bound only does something for `?Sized`
    //~| ERROR: relaxing a default bound only does something for `?Sized`
    ()
}

pub fn main() {}
