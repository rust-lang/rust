//! Regression test for: https://github.com/rust-lang/rust/issues/156837
//@ compile-flags: -Znext-solver=globally --crate-type=lib
//@ edition: 2018

#![warn(rust_2021_incompatible_closure_captures)]

async fn get() {}

fn check() {
    let mut v = get();
    (|| match v {
        (1, _) => (),
        //~^ ERROR mismatched types
    })()
}
