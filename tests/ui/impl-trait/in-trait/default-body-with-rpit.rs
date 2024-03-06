//@ edition:2021
//@ check-pass

#![allow(incomplete_features)]

use std::fmt::Debug;

trait Foo {
    #[allow(async_fn_in_trait)]
    async fn baz(&self) -> impl Debug {
        ""
    }
}

struct Bar;

impl Foo for Bar {}

fn main() {
    let _ = Bar.baz();
}
