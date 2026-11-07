//@ incremental
//@ edition: 2021
//@ revisions: assumptions no_assumptions
//@[assumptions] compile-flags: -Zhigher-ranked-assumptions
//@[assumptions] check-pass
//@[no_assumptions] known-bug: #110338

use std::future::*;
use std::marker::PhantomData;
use std::pin::Pin;
use std::task::*;

fn send<T: Send>(_: T) {}

pub trait Stream {
    type Item;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>>;
}

struct Empty<T>(PhantomData<fn() -> T>);

impl<T> Stream for Empty<T> {
    type Item = T;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        todo!()
    }
}

pub trait FnOnce1<A> {
    type Output;
    fn call_once(self, arg: A) -> Self::Output;
}

impl<T, A, R> FnOnce1<A> for T
where
    T: FnOnce(A) -> R,
{
    type Output = R;
    fn call_once(self, arg: A) -> R {
        self(arg)
    }
}

pub trait FnMut1<A>: FnOnce1<A> {
    fn call_mut(&mut self, arg: A) -> Self::Output;
}

impl<T, A, R> FnMut1<A> for T
where
    T: FnMut(A) -> R,
{
    fn call_mut(&mut self, arg: A) -> R {
        self(arg)
    }
}

struct Map<St, F>(St, F);

impl<St, F> Stream for Map<St, F>
where
    St: Stream,
    F: FnMut1<St::Item>,
{
    type Item = F::Output;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        todo!()
    }
}

struct FuturesOrdered<T: Future>(PhantomData<fn() -> T::Output>);

pub struct Buffered<St: Stream>(St, FuturesOrdered<St::Item>, usize)
where
    St::Item: Future;

impl<St> Stream for Buffered<St>
where
    St: Stream,
    St::Item: Future,
{
    type Item = <St::Item as Future>::Output;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        todo!()
    }
}

struct Next<'a, T: ?Sized>(&'a T);

impl<St: ?Sized + Stream + Unpin> Future for Next<'_, St> {
    type Output = Option<St::Item>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        todo!()
    }
}

fn main() {
    send(async {
        Next(&Buffered(Map(Empty(PhantomData), ready::<&()>), FuturesOrdered(PhantomData), 0)).await
    });
}
