// compile-flags: -Ztrait-solver=next
// check-pass
// edition:2021

#![feature(async_fn_in_trait)]

trait Foo {
    async fn bar() {}
}

fn main() {}
