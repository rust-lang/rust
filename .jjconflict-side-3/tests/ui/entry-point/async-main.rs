//! Regression test for https://github.com/rust-lang/rust/issues/78905.

//@ edition:2018

async fn main() {}
//~^ ERROR `main` function is not allowed to be `async`
