// edition:2021
// [next] compile-flags: -Zlower-impl-trait-in-trait-to-assoc-ty
// revisions: current next

#![feature(return_position_impl_trait_in_trait)]
#![allow(incomplete_features)]

use std::future::Future;

trait Captures<'a> {}
impl<T> Captures<'_> for T {}

trait Captures2<'a, 'b> {}
impl<T> Captures2<'_, '_> for T {}

pub trait AsyncTrait {
    fn async_fn(&self, buff: &[u8]) -> impl Future<Output = Vec<u8>>;
    fn async_fn_early<'a: 'a>(&self, buff: &'a [u8]) -> impl Future<Output = Vec<u8>>;
    fn async_fn_multiple<'a>(&'a self, buff: &[u8])
        -> impl Future<Output = Vec<u8>> + Captures<'a>;
    fn async_fn_reduce_outlive<'a, T>(
        &'a self,
        buff: &[u8],
        t: T,
    ) -> impl Future<Output = Vec<u8>> + 'a;
    fn async_fn_reduce<'a, T>(
        &'a self,
        buff: &[u8],
        t: T,
    ) -> impl Future<Output = Vec<u8>> + Captures<'a>;
}

pub struct Struct;

impl AsyncTrait for Struct {
    fn async_fn<'a>(&self, buff: &'a [u8]) -> impl Future<Output = Vec<u8>> + 'a {
        //~^ ERROR return type captures more lifetimes than trait definition
        async move { buff.to_vec() }
    }

    fn async_fn_early<'a: 'a>(&self, buff: &'a [u8]) -> impl Future<Output = Vec<u8>> + 'a {
        //~^ ERROR return type captures more lifetimes than trait definition
        async move { buff.to_vec() }
    }

    fn async_fn_multiple<'a, 'b>(
        &'a self,
        buff: &'b [u8],
    ) -> impl Future<Output = Vec<u8>> + Captures2<'a, 'b> {
        //~^ ERROR return type captures more lifetimes than trait definition
        async move { buff.to_vec() }
    }

    fn async_fn_reduce_outlive<'a, 'b, T>(
        &'a self,
        buff: &'b [u8],
        t: T,
    ) -> impl Future<Output = Vec<u8>> {
        //~^ ERROR the parameter type `T` may not live long enough
        async move {
            let _t = t;
            vec![]
        }
    }

    // OK: We remove the `Captures<'a>`, providing a guarantee that we don't capture `'a`,
    // but we still fulfill the `Captures<'a>` trait bound.
    fn async_fn_reduce<'a, 'b, T>(&'a self, buff: &'b [u8], t: T) -> impl Future<Output = Vec<u8>> {
        async move {
            let _t = t;
            vec![]
        }
    }
}

fn main() {}
