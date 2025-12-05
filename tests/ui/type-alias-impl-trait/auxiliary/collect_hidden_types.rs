#![feature(impl_trait_in_assoc_type)]

//@ edition:2018

use std::future::Future;

pub trait Service<Request> {
    type Future: Future<Output = ()>;
    fn call(&mut self, req: Request) -> Self::Future;
}

// NOTE: the pub(crate) here is critical
pub(crate) fn new() -> () {}

pub struct A;
impl Service<()> for A {
    type Future = impl Future<Output = ()>;
    fn call(&mut self, _: ()) -> Self::Future {
        async { new() }
    }
}
