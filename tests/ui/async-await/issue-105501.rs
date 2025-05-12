//@ check-pass
//@ edition:2018

// This is a regression test for https://github.com/rust-lang/rust/issues/105501.
// It was minified from the published `msf-ice:0.2.1` crate which failed in a crater run.
// A faulty compiler was triggering a `higher-ranked lifetime error`:
//
// > could not prove `[async block@...]: Send`

use mini_futures::Stream;

fn is_send(_: impl Send) {}

pub fn main() {
    let fut = async {
        let mut stream = mini_futures::iter([()])
            .then(|_| async {})
            .map(|_| async { None })
            .buffered()
            .filter_map(std::future::ready);

        stream.next().await
    };

    is_send(async move {
        let _: Option<()> = fut.await;
    });
}

// this is a simplified subset of `futures::StreamExt` and related types
mod mini_futures {
    use std::future::Future;
    use std::pin::Pin;
    use std::task::{Context, Poll};

    pub fn iter<I>(_: I) -> Iter<I::IntoIter>
    where
        I: IntoIterator,
    {
        todo!()
    }

    pub trait Stream {
        type Item;

        fn then<Fut, F>(self, _: F) -> Then<Self, Fut, F>
        where
            F: FnMut(Self::Item) -> Fut,
            Fut: Future,
            Self: Sized,
        {
            todo!()
        }

        fn map<T, F>(self, _: F) -> Map<Self, F>
        where
            F: FnMut(Self::Item) -> T,
            Self: Sized,
        {
            todo!()
        }

        fn buffered(self) -> Buffered<Self>
        where
            Self::Item: Future,
            Self: Sized,
        {
            todo!()
        }

        fn filter_map<Fut, T, F>(self, _: F) -> FilterMap<Self, Fut, F>
        where
            F: FnMut(Self::Item) -> Fut,
            Fut: Future<Output = Option<T>>,
            Self: Sized,
        {
            todo!()
        }

        fn next(&mut self) -> Next<'_, Self> {
            todo!()
        }
    }

    pub struct Iter<I> {
        __: I,
    }
    impl<I> Stream for Iter<I>
    where
        I: Iterator,
    {
        type Item = I::Item;
    }

    pub struct Then<St, Fut, F> {
        __: (St, Fut, F),
    }
    impl<St, Fut, F> Stream for Then<St, Fut, F>
    where
        St: Stream,
        F: FnMut(St::Item) -> Fut,
        Fut: Future,
    {
        type Item = Fut::Output;
    }

    pub struct Map<St, F> {
        __: (St, F),
    }
    impl<St, F> Stream for Map<St, F>
    where
        St: Stream,
        F: FnMut1<St::Item>,
    {
        type Item = F::Output;
    }

    pub trait FnMut1<A> {
        type Output;
    }
    impl<T, A, R> FnMut1<A> for T
    where
        T: FnMut(A) -> R,
    {
        type Output = R;
    }

    pub struct Buffered<St>
    where
        St: Stream,
        St::Item: Future,
    {
        __: (St, St::Item),
    }
    impl<St> Stream for Buffered<St>
    where
        St: Stream,
        St::Item: Future,
    {
        type Item = <St::Item as Future>::Output;
    }

    pub struct FilterMap<St, Fut, F> {
        __: (St, Fut, F),
    }
    impl<St, Fut, F, T> Stream for FilterMap<St, Fut, F>
    where
        St: Stream,
        F: FnMut1<St::Item, Output = Fut>,
        Fut: Future<Output = Option<T>>,
    {
        type Item = T;
    }

    pub struct Next<'a, St: ?Sized> {
        __: &'a mut St,
    }
    impl<St: ?Sized + Stream> Future for Next<'_, St> {
        type Output = Option<St::Item>;

        fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
            todo!()
        }
    }
}
