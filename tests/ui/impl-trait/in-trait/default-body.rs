// check-pass
// edition:2021

#![feature(async_fn_in_trait, return_position_impl_trait_in_trait)]
#![allow(incomplete_features)]

use std::fmt::Debug;

trait Foo {
    async fn baz(&self) -> &str {
        ""
    }
}

struct Bar;

impl Foo for Bar {}

fn main() {
    let _ = Bar.baz();
}
