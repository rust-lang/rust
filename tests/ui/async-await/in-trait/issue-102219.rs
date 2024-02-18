//@ compile-flags:--crate-type=lib
//@ edition:2021
//@ check-pass

#![allow(incomplete_features)]

trait T {
    #[allow(async_fn_in_trait)]
    async fn foo();
}
