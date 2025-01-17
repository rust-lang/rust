//@ run-pass
//@ revisions: default feature
#![cfg_attr(feature, feature(arbitrary_self_types))]

use std::pin::Pin;
use std::ops::DerefMut;
use std::marker::Unpin;

struct TryChunks;

impl TryChunks {
    #[allow(dead_code)]
    fn take(self: std::pin::Pin<&mut Self>) -> usize {
        1
    }
}

#[allow(dead_code)]
trait Stream {
    fn poll_next(self: std::pin::Pin<&mut Self>);
}

#[allow(dead_code)]
trait StreamExt: Stream {
    #[allow(dead_code)]
    fn take(self) -> usize where Self: Sized
    {
        2
    }
}

impl<T: ?Sized> StreamExt for T where T: Stream {}

impl Stream for TryChunks {
    fn poll_next(self: std::pin::Pin<&mut Self>) {
        assert_eq!(self.take(), 1);
    }
}

#[allow(dead_code)]
impl<S: ?Sized + Stream + Unpin> Stream for &mut S {
    #[allow(dead_code)]
    fn poll_next(mut self: Pin<&mut Self>)  {
        S::poll_next(Pin::new(&mut **self))
    }
}

#[allow(dead_code)]
impl<P> Stream for Pin<P>
where
    P: DerefMut + Unpin,
    P::Target: Stream,
{
    #[allow(dead_code)]
    fn poll_next(self: Pin<&mut Self>) {
        self.get_mut().as_mut().poll_next()
    }
}

fn main() {
    let mut item = Box::pin(TryChunks);
    item.as_mut().poll_next();
}
