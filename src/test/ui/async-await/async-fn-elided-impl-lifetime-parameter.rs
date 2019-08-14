// Check that `async fn` inside of an impl with `'_`
// in the header compiles correctly.
//
// Regression test for #63500.
//
// check-pass
// edition:2018

#![feature(async_await)]

struct Foo<'a>(&'a u8);

impl Foo<'_> {
    async fn bar() {}
}

fn main() { }
