//@ revisions: rpass1 rpass2
//@ edition:2021

// See https://github.com/rust-lang/rust/issues/98890

#![allow(unused)]

struct Foo;

impl Foo {
    async fn f(&self, _: &&()) -> &() {
        &()
    }
}

#[cfg(rpass2)]
enum Bar {}

fn main() {}
