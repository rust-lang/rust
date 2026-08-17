// Regression test for https://github.com/rust-lang/rust/issues/159128.

#![feature(fn_delegation)]

trait Trait {
    fn method(self);
}

struct S<S>;
//~^ ERROR type parameter `S` is never used

impl<T> Trait for S<T> {
    reuse Trait::method { S }
    //~^ ERROR type annotations needed
}

fn main() {}
