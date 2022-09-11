// check-pass
// edition:2021

#![feature(return_position_impl_trait_in_trait)]
#![allow(incomplete_features)]

use std::fmt::Debug;

trait Foo {
    async fn baz() -> impl Debug {
        Self::baz().await
    }
}

fn main() {}
