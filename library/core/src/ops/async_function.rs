use crate::future::Future;
use crate::marker::Tuple;

/// An async-aware version of the [`Fn`](crate::ops::Fn) trait.
///
/// All `async fn` and functions returning futures implement this trait.
#[unstable(feature = "async_fn_traits", issue = "none")]
#[rustc_paren_sugar]
#[fundamental]
#[must_use = "async closures are lazy and do nothing unless called"]
pub trait AsyncFn<Args: Tuple>: AsyncFnMut<Args> {
    /// Future returned by [`AsyncFn::async_call`].
    #[unstable(feature = "async_fn_traits", issue = "none")]
    type CallFuture<'a>: Future<Output = Self::Output>
    where
        Self: 'a;

    /// Call the [`AsyncFn`], returning a future which may borrow from the called closure.
    #[unstable(feature = "async_fn_traits", issue = "none")]
    extern "rust-call" fn async_call(&self, args: Args) -> Self::CallFuture<'_>;
}

/// An async-aware version of the [`FnMut`](crate::ops::FnMut) trait.
///
/// All `async fn` and functions returning futures implement this trait.
#[unstable(feature = "async_fn_traits", issue = "none")]
#[rustc_paren_sugar]
#[fundamental]
#[must_use = "async closures are lazy and do nothing unless called"]
pub trait AsyncFnMut<Args: Tuple>: AsyncFnOnce<Args> {
    /// Future returned by [`AsyncFnMut::async_call_mut`].
    #[unstable(feature = "async_fn_traits", issue = "none")]
    type CallMutFuture<'a>: Future<Output = Self::Output>
    where
        Self: 'a;

    /// Call the [`AsyncFnMut`], returning a future which may borrow from the called closure.
    #[unstable(feature = "async_fn_traits", issue = "none")]
    extern "rust-call" fn async_call_mut(&mut self, args: Args) -> Self::CallMutFuture<'_>;
}

/// An async-aware version of the [`FnOnce`](crate::ops::FnOnce) trait.
///
/// All `async fn` and functions returning futures implement this trait.
#[unstable(feature = "async_fn_traits", issue = "none")]
#[rustc_paren_sugar]
#[fundamental]
#[must_use = "async closures are lazy and do nothing unless called"]
pub trait AsyncFnOnce<Args: Tuple> {
    /// Future returned by [`AsyncFnOnce::async_call_once`].
    #[unstable(feature = "async_fn_traits", issue = "none")]
    type CallOnceFuture: Future<Output = Self::Output>;

    /// Output type of the called closure's future.
    #[unstable(feature = "async_fn_traits", issue = "none")]
    type Output;

    /// Call the [`AsyncFnOnce`], returning a future which may move out of the called closure.
    #[unstable(feature = "async_fn_traits", issue = "none")]
    extern "rust-call" fn async_call_once(self, args: Args) -> Self::CallOnceFuture;
}

mod impls {
    use super::{AsyncFn, AsyncFnMut, AsyncFnOnce};
    use crate::future::Future;
    use crate::marker::Tuple;

    #[unstable(feature = "async_fn_traits", issue = "none")]
    impl<F: Fn<A>, A: Tuple> AsyncFn<A> for F
    where
        <F as FnOnce<A>>::Output: Future,
    {
        type CallFuture<'a> = impl Future<Output = Self::Output> where Self: 'a;

        extern "rust-call" fn async_call(&self, args: A) -> Self::CallFuture<'_> {
            async { self.call(args).await }
        }
    }

    #[unstable(feature = "async_fn_traits", issue = "none")]
    impl<F: FnMut<A>, A: Tuple> AsyncFnMut<A> for F
    where
        <F as FnOnce<A>>::Output: Future,
    {
        type CallMutFuture<'a> = impl Future<Output = Self::Output> where Self: 'a;

        extern "rust-call" fn async_call_mut(&mut self, args: A) -> Self::CallMutFuture<'_> {
            async { self.call_mut(args).await }
        }
    }

    #[unstable(feature = "async_fn_traits", issue = "none")]
    impl<F: FnOnce<A>, A: Tuple> AsyncFnOnce<A> for F
    where
        <F as FnOnce<A>>::Output: Future,
    {
        type CallOnceFuture = impl Future<Output = Self::Output>;

        type Output = <<F as FnOnce<A>>::Output as Future>::Output;

        extern "rust-call" fn async_call_once(self, args: A) -> Self::CallOnceFuture {
            async { self.call_once(args).await }
        }
    }
}
