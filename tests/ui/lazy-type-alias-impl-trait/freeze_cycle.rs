//@ check-pass

#![feature(coroutine_trait)]

use std::future::Future;
use std::ops::{Coroutine, CoroutineState};
use std::pin::Pin;
use std::ptr::NonNull;
use std::task::{Context, Poll};

fn main() {}

#[derive(Debug, Copy, Clone)]
pub struct ResumeTy(NonNull<Context<'static>>);

unsafe impl Send for ResumeTy {}

unsafe impl Sync for ResumeTy {}

pub const fn from_coroutine<T>(gen: T) -> impl Future<Output = T::Return>
where
    T: Coroutine<ResumeTy, Yield = ()>,
{
    struct GenFuture<T: Coroutine<ResumeTy, Yield = ()>>(T);

    impl<T: Coroutine<ResumeTy, Yield = ()>> Future for GenFuture<T> {
        type Output = T::Return;
        fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
            // SAFETY: Safe because we're !Unpin + !Drop, and this is just a field projection.
            let gen = unsafe { Pin::map_unchecked_mut(self, |s| &mut s.0) };

            // Resume the coroutine, turning the `&mut Context` into a `NonNull` raw pointer. The
            // `.await` lowering will safely cast that back to a `&mut Context`.
            match gen.resume(ResumeTy(NonNull::from(cx).cast::<Context<'static>>())) {
                CoroutineState::Yielded(()) => Poll::Pending,
                CoroutineState::Complete(x) => Poll::Ready(x),
            }
        }
    }

    GenFuture(gen)
}
