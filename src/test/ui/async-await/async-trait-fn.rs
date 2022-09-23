// edition:2018
// check-pass

#![feature(async_fn_in_trait)] //~ WARN the feature `async_fn_in_trait` is incomplete

trait T {
    async fn foo();
    // async fn bar(&self);
}

struct Impl;

impl T for Impl {
    async fn foo() {}
}

fn main() {}
