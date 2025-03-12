//@ compile-flags:--crate-type=lib
//@ edition:2021
//@ check-pass

trait T {
    async fn foo();
}
