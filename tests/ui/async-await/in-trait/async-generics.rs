// check-fail
// known-bug: #102682
// edition: 2021
// ignore-compare-mode-lower-impl-trait-in-trait-to-assoc-ty

#![feature(async_fn_in_trait)]
#![allow(incomplete_features)]

trait MyTrait<T, U> {
    async fn foo(&self) -> &(T, U);
}

impl<T, U> MyTrait<T, U> for (T, U) {
    async fn foo(&self) -> &(T, U) {
        self
    }
}

fn main() {}
