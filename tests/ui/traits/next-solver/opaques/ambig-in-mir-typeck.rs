// Regression test for #132335. This previously ICE'd due to ambiguity
// in MIR typeck.

//@ compile-flags: -Znext-solver=globally --crate-type lib
//@ edition: 2018
//@ check-pass
use core::future::Future;
use core::pin::Pin;

pub trait Unit {}
impl Unit for () {}

pub fn get_all_files_in_dir() -> Pin<Box<dyn Future<Output = impl Unit>>> {
    Box::pin(async {
        get_all_files_in_dir().await;
    })
}
