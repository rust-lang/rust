//! Lang item adapters to translate a bare coroutine into a `Future`, `AsyncIterator`, or `Iterator`.
//! Functions in this file are supposed to be used by the compiler only.

#![allow(unreachable_pub)]

use crate::async_iter::AsyncIterator;
use crate::future::{ResumeTy, wrap_context};
use crate::iter::{FusedIterator, Iterator};
use crate::ops::coroutine::{Coroutine, CoroutineState};
use crate::pin::Pin;
use crate::task::{Context, Poll};

#[inline(always)]
#[doc(hidden)]
#[unstable(feature = "gen_future", issue = "none")]
#[rustc_const_unstable(feature = "gen_future", issue = "none")]
#[rustc_const_stable_indirect] // Avoid emitting a diagnostic for unstable const fn.
#[lang = "iter_from_coroutine"]
pub const fn iter_from_coroutine<C: Coroutine<Return = ()> + Unpin>(
    coroutine: C,
) -> CoroutineIterator<C> {
    CoroutineIterator(coroutine)
}

#[doc(hidden)]
#[unstable(feature = "gen_future", issue = "none")]
#[lang = "CoroutineIterator"]
#[derive(Clone)]
pub struct CoroutineIterator<C>(C);

#[unstable(feature = "gen_future", issue = "none")]
impl<C: Coroutine<Return = ()> + Unpin> Iterator for CoroutineIterator<C> {
    type Item = C::Yield;

    #[inline(always)]
    #[track_caller]
    fn next(&mut self) -> Option<Self::Item> {
        match Pin::new(&mut self.0).resume(()) {
            CoroutineState::Yielded(n) => Some(n),
            CoroutineState::Complete(()) => None,
        }
    }
}

#[unstable(feature = "gen_future", issue = "none")]
impl<C: Coroutine<Return = ()> + Unpin> FusedIterator for CoroutineIterator<C> {}

#[inline(always)]
#[doc(hidden)]
#[unstable(feature = "gen_future", issue = "none")]
#[rustc_const_unstable(feature = "gen_future", issue = "none")]
#[rustc_const_stable_indirect] // Avoid emitting a diagnostic for unstable const fn.
#[lang = "future_from_coroutine"]
pub const fn future_from_coroutine<C: Coroutine<ResumeTy, Yield = ()>>(
    coroutine: C,
) -> CoroutineFuture<C> {
    CoroutineFuture(coroutine)
}

#[unstable(feature = "gen_future", issue = "none")]
#[lang = "CoroutineFuture"]
pub struct CoroutineFuture<C>(C);

#[unstable(feature = "gen_future", issue = "none")]
impl<C> Future for CoroutineFuture<C>
where
    C: Coroutine<ResumeTy, Yield = ()>,
{
    type Output = C::Return;

    #[inline(always)]
    #[track_caller]
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // SAFETY: `self.0` is pinned when `self` is.
        let inner = unsafe { self.map_unchecked_mut(|c| &mut c.0) };
        // SAFETY: `wrap_context` creates a `ResumeTy` from a valid mutable reference to `Context`.
        // This is only used inside the compiler-generated `resume` method.
        let cx = unsafe { wrap_context(cx) };
        match inner.resume(cx) {
            CoroutineState::Complete(val) => Poll::Ready(val),
            CoroutineState::Yielded(()) => Poll::Pending,
        }
    }
}

#[doc(hidden)]
#[unstable(feature = "gen_future", issue = "none")]
#[rustc_const_unstable(feature = "gen_future", issue = "none")]
#[rustc_const_stable_indirect] // Avoid emitting a diagnostic for unstable const fn.
#[lang = "async_iterator_from_coroutine"]
pub const fn async_iterator_from_coroutine<
    C: Coroutine<ResumeTy, Yield = Poll<Option<I>>, Return = ()>,
    I,
>(
    coroutine: C,
) -> CoroutineAsyncIterator<C> {
    CoroutineAsyncIterator(coroutine)
}

#[unstable(feature = "gen_future", issue = "none")]
pub struct CoroutineAsyncIterator<C>(C);

impl<C, I> AsyncIterator for CoroutineAsyncIterator<C>
where
    C: Coroutine<ResumeTy, Yield = Poll<Option<I>>, Return = ()>,
{
    type Item = I;

    #[inline(always)]
    #[track_caller]
    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // SAFETY: `self.0` is pinned when `self` is.
        let inner = unsafe { self.map_unchecked_mut(|c| &mut c.0) };
        // SAFETY: `wrap_context` creates a `ResumeTy` from a valid mutable reference to `Context`.
        // This is only used inside the compiler-generated `resume` method.
        let cx = unsafe { wrap_context(cx) };
        match inner.resume(cx) {
            CoroutineState::Complete(()) => Poll::Ready(None),
            CoroutineState::Yielded(yielded) => yielded,
        }
    }
}
