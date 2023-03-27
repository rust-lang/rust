// compile-flags:--crate-type=lib
// edition:2021
// check-pass
// [next] compile-flags: -Zlower-impl-trait-in-trait-to-assoc-ty
// revisions: current next

#![feature(async_fn_in_trait)]
#![allow(incomplete_features)]

trait T {
    async fn foo();
}
