//@ check-pass
//@ edition:2021

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
