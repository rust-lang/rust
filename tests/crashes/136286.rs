//@ known-bug: #136286
//@ compile-flags: --edition=2024

#![feature(async_fn_in_dyn_trait)]
trait A {
    async fn b(self: A);
}
