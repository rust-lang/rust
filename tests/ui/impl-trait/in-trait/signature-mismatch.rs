// edition:2021
// [next] compile-flags: -Zlower-impl-trait-in-trait-to-assoc-ty
// revisions: current next

#![feature(return_position_impl_trait_in_trait)]
#![allow(incomplete_features)]

use std::future::Future;

pub trait AsyncTrait {
    fn async_fn(&self, buff: &[u8]) -> impl Future<Output = Vec<u8>>;
}

pub struct Struct;

impl AsyncTrait for Struct {
    fn async_fn<'a>(&self, buff: &'a [u8]) -> impl Future<Output = Vec<u8>> + 'a {
        //~^ ERROR `impl` item signature doesn't match `trait` item signature
        async move { buff.to_vec() }
    }
}

fn main() {}
