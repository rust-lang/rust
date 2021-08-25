use crate::ops::DerefMut;
use crate::pin::Pin;
use crate::task::{Context, Poll};

/// An interface for dealing with asynchronous iterators.
///
/// This is the main stream trait. For more about the concept of streams
/// generally, please see the [module-level documentation]. In particular, you
/// may want to know how to [implement `Stream`][impl].
///
/// [module-level documentation]: index.html
/// [impl]: index.html#implementing-stream
#[unstable(feature = "async_stream", issue = "79024")]
#[must_use = "streams do nothing unless polled"]
pub trait Stream {
    /// The type of items yielded by the stream.
    type Item;

    /// Attempt to pull out the next value of this stream, registering the
    /// current task for wakeup if the value is not yet available, and returning
    /// `None` if the stream is exhausted.
    ///
    /// # Return value
    ///
    /// There are several possible return values, each indicating a distinct
    /// stream state:
    ///
    /// - `Poll::Pending` means that this stream's next value is not ready
    /// yet. Implementations will ensure that the current task will be notified
    /// when the next value may be ready.
    ///
    /// - `Poll::Ready(Some(val))` means that the stream has successfully
    /// produced a value, `val`, and may produce further values on subsequent
    /// `poll_next` calls.
    ///
    /// - `Poll::Ready(None)` means that the stream has terminated, and
    /// `poll_next` should not be invoked again.
    ///
    /// # Panics
    ///
    /// Once a stream has finished (returned `Ready(None)` from `poll_next`), calling its
    /// `poll_next` method again may panic, block forever, or cause other kinds of
    /// problems; the `Stream` trait places no requirements on the effects of
    /// such a call. However, as the `poll_next` method is not marked `unsafe`,
    /// Rust's usual rules apply: calls must never cause undefined behavior
    /// (memory corruption, incorrect use of `unsafe` functions, or the like),
    /// regardless of the stream's state.
    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>>;

    /// Returns the bounds on the remaining length of the stream.
    ///
    /// Specifically, `size_hint()` returns a tuple where the first element
    /// is the lower bound, and the second element is the upper bound.
    ///
    /// The second half of the tuple that is returned is an <code>[Option]<[usize]></code>.
    /// A [`None`] here means that either there is no known upper bound, or the
    /// upper bound is larger than [`usize`].
    ///
    /// # Implementation notes
    ///
    /// It is not enforced that a stream implementation yields the declared
    /// number of elements. A buggy stream may yield less than the lower bound
    /// or more than the upper bound of elements.
    ///
    /// `size_hint()` is primarily intended to be used for optimizations such as
    /// reserving space for the elements of the stream, but must not be
    /// trusted to e.g., omit bounds checks in unsafe code. An incorrect
    /// implementation of `size_hint()` should not lead to memory safety
    /// violations.
    ///
    /// That said, the implementation should provide a correct estimation,
    /// because otherwise it would be a violation of the trait's protocol.
    ///
    /// The default implementation returns <code>(0, [None])</code> which is correct for any
    /// stream.
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, None)
    }
}

#[unstable(feature = "async_stream", issue = "79024")]
impl<S: ?Sized + Stream + Unpin> Stream for &mut S {
    type Item = S::Item;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        S::poll_next(Pin::new(&mut **self), cx)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (**self).size_hint()
    }
}

#[unstable(feature = "async_stream", issue = "79024")]
impl<P> Stream for Pin<P>
where
    P: DerefMut + Unpin,
    P::Target: Stream,
{
    type Item = <P::Target as Stream>::Item;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.get_mut().as_mut().poll_next(cx)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (**self).size_hint()
    }
}
