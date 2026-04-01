//@ check-pass

use std::error::Error;
use std::pin::Pin;
use std::task::{Context, Poll};

pub trait Stream {
    type Item;
    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>>;
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, None)
    }
}

pub trait TryStream: Stream {
    type Ok;
    type Error;

    fn try_poll_next(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Self::Ok, Self::Error>>>;
}

impl<S, T, E> TryStream for S
where
    S: ?Sized + Stream<Item = Result<T, E>>,
{
    type Ok = T;
    type Error = E;

    fn try_poll_next(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Self::Ok, Self::Error>>> {
        self.poll_next(cx)
    }
}

pub trait ServerSentEvent: Sized + Send + Sync + 'static {}

impl<T: Send + Sync + 'static> ServerSentEvent for T {}

struct SseKeepAlive<S> {
    event_stream: S,
}

struct SseComment<T>(T);

impl<S> Stream for SseKeepAlive<S>
where
    S: TryStream + Send + 'static,
    S::Ok: ServerSentEvent,
    S::Error: Error + Send + Sync + 'static,
{
    type Item = Result<SseComment<&'static str>, ()>;
    fn poll_next(self: Pin<&mut Self>, _cx: &mut Context) -> Poll<Option<Self::Item>> {
        unimplemented!()
    }
}

pub fn keep<S>(
    event_stream: S,
) -> impl TryStream<Ok = impl ServerSentEvent + Send + 'static, Error = ()> + Send + 'static
where
    S: TryStream + Send + 'static,
    S::Ok: ServerSentEvent + Send,
    S::Error: Error + Send + Sync + 'static,
{
    SseKeepAlive { event_stream }
}

fn main() {}
