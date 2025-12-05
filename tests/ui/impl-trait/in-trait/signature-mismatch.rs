//@ edition:2021
//@ revisions: success failure
//@[success] check-pass

use std::future::Future;

pub trait Captures<'a> {}
impl<T> Captures<'_> for T {}

pub trait Captures2<'a, 'b> {}
impl<T> Captures2<'_, '_> for T {}

pub trait AsyncTrait {
    #[cfg(success)]
    fn async_fn(&self, buff: &[u8]) -> impl Future<Output = Vec<u8>>;

    #[cfg(success)]
    fn async_fn_early<'a: 'a>(&self, buff: &'a [u8]) -> impl Future<Output = Vec<u8>>;

    #[cfg(success)]
    fn async_fn_multiple<'a>(&'a self, buff: &[u8])
    -> impl Future<Output = Vec<u8>> + Captures<'a>;

    #[cfg(failure)]
    fn async_fn_reduce_outlive<'a, T>(
        &'a self,
        buff: &[u8],
        t: T,
    ) -> impl Future<Output = Vec<u8>> + 'a;

    #[cfg(success)]
    fn async_fn_reduce<'a, T>(
        &'a self,
        buff: &[u8],
        t: T,
    ) -> impl Future<Output = Vec<u8>> + Captures<'a>;
}

pub struct Struct;

impl AsyncTrait for Struct {
    // Does not capture more lifetimes that trait def'n, since trait def'n
    // implicitly captures all in-scope lifetimes.
    #[cfg(success)]
    #[expect(refining_impl_trait)]
    fn async_fn<'a>(&self, buff: &'a [u8]) -> impl Future<Output = Vec<u8>> + 'a {
        async move { buff.to_vec() }
    }

    // Does not capture more lifetimes that trait def'n, since trait def'n
    // implicitly captures all in-scope lifetimes.
    #[cfg(success)]
    #[expect(refining_impl_trait)]
    fn async_fn_early<'a: 'a>(&self, buff: &'a [u8]) -> impl Future<Output = Vec<u8>> + 'a {
        async move { buff.to_vec() }
    }

    // Does not capture more lifetimes that trait def'n, since trait def'n
    // implicitly captures all in-scope lifetimes.
    #[cfg(success)]
    #[expect(refining_impl_trait)]
    fn async_fn_multiple<'a, 'b>(
        &'a self,
        buff: &'b [u8],
    ) -> impl Future<Output = Vec<u8>> + Captures2<'a, 'b> {
        async move { buff.to_vec() }
    }

    // This error message is awkward, but `impl Future<Output = Vec<u8>>`
    // cannot outlive `'a` (from the trait signature) because it captures
    // both `T` and `'b`.
    #[cfg(failure)]
    fn async_fn_reduce_outlive<'a, 'b, T>(
        &'a self,
        buff: &'b [u8],
        t: T,
    ) -> impl Future<Output = Vec<u8>> {
        //[failure]~^ ERROR the type `impl Future<Output = Vec<u8>>` does not fulfill the required lifetime
        async move {
            let _t = t;
            vec![]
        }
    }

    // Does not capture fewer lifetimes that trait def'n (not that it matters),
    // since impl also captures all in-scope lifetimes.
    #[cfg(success)]
    fn async_fn_reduce<'a, 'b, T>(&'a self, buff: &'b [u8], t: T) -> impl Future<Output = Vec<u8>> {
        async move {
            let _t = t;
            vec![]
        }
    }
}

fn main() {}
