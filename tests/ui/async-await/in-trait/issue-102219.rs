// compile-flags:--crate-type=lib
// edition:2021
// check-pass

#![feature(async_fn_in_trait)]
#![allow(incomplete_features)]

trait T {
    async fn foo();
}
