//@ edition:2021
//@ check-pass
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

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
