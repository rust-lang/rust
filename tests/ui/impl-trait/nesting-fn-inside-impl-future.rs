//! Regression test for <https://github.com/rust-lang/rust/issues/165091>

//@ check-pass
//@ edition: 2024

#![deny(refining_impl_trait)]

use std::future::Future;

trait Foo {
    fn bar(&self) -> impl Future<Output = Box<dyn Fn(&mut ())>>;
}

struct Nope;
impl Foo for Nope {
    fn bar(&self) -> impl Future<Output = Box<dyn Fn(&mut ())>> {
        async { todo!() }
    }
}

fn main() {}
