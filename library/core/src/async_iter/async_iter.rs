use crate::ops::DerefMut;
use crate::pin::Pin;
use crate::task::{Context, Poll};

/// A trait for dealing with asynchronous iterators.
///
/// This is the main async iterator trait. For more about the concept of async iterators
/// generally, please see the [module-level documentation]. In particular, you
/// may want to know how to [implement `AsyncIterator`][impl].
///
/// [module-level documentation]: index.html
/// [impl]: index.html#implementing-async-iterator
#[unstable(feature = "async_iterator", issue = "79024")]
#[must_use = "async iterators do nothing unless polled"]
#[doc(alias = "Stream")]
#[lang = "async_iterator"]
pub trait AsyncIterator {
    /// The type of items yielded by the async iterator.
    type Item;

    /// Attempts to pull out the next value of this async iterator, registering the
    /// current task for wakeup if the value is not yet available, and returning
    /// `None` if the async iterator is exhausted.
    ///
    /// # Return value
    ///
    /// There are several possible return values, each indicating a distinct
    /// async iterator state:
    ///
    /// - `Poll::Pending` means that this async iterator's next value is not ready
    /// yet. Implementations will ensure that the current task will be notified
    /// when the next value may be ready.
    ///
    /// - `Poll::Ready(Some(val))` means that the async iterator has successfully
    /// produced a value, `val`, and may produce further values on subsequent
    /// `poll_next` calls.
    ///
    /// - `Poll::Ready(None)` means that the async iterator has terminated, and
    /// `poll_next` should not be invoked again.
    ///
    /// # Panics
    ///
    /// Once an async iterator has finished (returned `Ready(None)` from `poll_next`), calling its
    /// `poll_next` method again may panic, block forever, or cause other kinds of
    /// problems; the `AsyncIterator` trait places no requirements on the effects of
    /// such a call. However, as the `poll_next` method is not marked `unsafe`,
    /// Rust's usual rules apply: calls must never cause undefined behavior
    /// (memory corruption, incorrect use of `unsafe` functions, or the like),
    /// regardless of the async iterator's state.
    #[lang = "async_iterator_poll_next"]
    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>>;

    /// Returns the bounds on the remaining length of the async iterator.
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
    /// It is not enforced that an async iterator implementation yields the declared
    /// number of elements. A buggy async iterator may yield less than the lower bound
    /// or more than the upper bound of elements.
    ///
    /// `size_hint()` is primarily intended to be used for optimizations such as
    /// reserving space for the elements of the async iterator, but must not be
    /// trusted to e.g., omit bounds checks in unsafe code. An incorrect
    /// implementation of `size_hint()` should not lead to memory safety
    /// violations.
    ///
    /// That said, the implementation should provide a correct estimation,
    /// because otherwise it would be a violation of the trait's protocol.
    ///
    /// The default implementation returns <code>(0, [None])</code> which is correct for any
    /// async iterator.
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, None)
    }
}

#[unstable(feature = "async_iterator", issue = "79024")]
impl<S: ?Sized + AsyncIterator + Unpin> AsyncIterator for &mut S {
    type Item = S::Item;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        S::poll_next(Pin::new(&mut **self), cx)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (**self).size_hint()
    }
}

#[unstable(feature = "async_iterator", issue = "79024")]
impl<P> AsyncIterator for Pin<P>
where
    P: DerefMut,
    P::Target: AsyncIterator,
{
    type Item = <P::Target as AsyncIterator>::Item;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        <P::Target as AsyncIterator>::poll_next(self.as_deref_mut(), cx)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (**self).size_hint()
    }
}

#[unstable(feature = "async_gen_internals", issue = "none")]
impl<T> Poll<Option<T>> {
    /// A helper function for internal desugaring -- produces `Ready(Some(t))`,
    /// which corresponds to the async iterator yielding a value.
    #[doc(hidden)]
    #[unstable(feature = "async_gen_internals", issue = "none")]
    #[lang = "AsyncGenReady"]
    pub fn async_gen_ready(t: T) -> Self {
        Poll::Ready(Some(t))
    }

    /// A helper constant for internal desugaring -- produces `Pending`,
    /// which corresponds to the async iterator pending on an `.await`.
    #[doc(hidden)]
    #[unstable(feature = "async_gen_internals", issue = "none")]
    #[lang = "AsyncGenPending"]
    // FIXME(gen_blocks): This probably could be deduplicated.
    pub const PENDING: Self = Poll::Pending;

    /// A helper constant for internal desugaring -- produces `Ready(None)`,
    /// which corresponds to the async iterator finishing its iteration.
    #[doc(hidden)]
    #[unstable(feature = "async_gen_internals", issue = "none")]
    #[lang = "AsyncGenFinished"]
    pub const FINISHED: Self = Poll::Ready(None);
}

/// Converts something into an async iterator
#[unstable(feature = "async_iterator", issue = "79024")]
pub trait IntoAsyncIterator {
    /// The type of the item yielded by the iterator
    type Item;
    /// The type of the resulting iterator
    type IntoAsyncIter: AsyncIterator<Item = Self::Item>;

    /// Converts `self` into an async iterator
    #[lang = "into_async_iter_into_iter"]
    fn into_async_iter(self) -> Self::IntoAsyncIter;
}

#[unstable(feature = "async_iterator", issue = "79024")]
impl<I: AsyncIterator> IntoAsyncIterator for I {
    type Item = I::Item;
    type IntoAsyncIter = I;

    fn into_async_iter(self) -> Self::IntoAsyncIter {
        self
    }
}
