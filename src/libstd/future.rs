//! Asynchronous values.

use core::cell::Cell;
use core::marker::Unpin;
use core::pin::Pin;
use core::option::Option;
use core::ptr::NonNull;
use core::task::{LocalWaker, Poll};
use core::ops::{Drop, Generator, GeneratorState};

#[doc(inline)]
pub use core::future::*;

/// Wrap a future in a generator.
///
/// This function returns a `GenFuture` underneath, but hides it in `impl Trait` to give
/// better error messages (`impl Future` rather than `GenFuture<[closure.....]>`).
#[unstable(feature = "gen_future", issue = "50547")]
pub fn from_generator<T: Generator<Yield = ()>>(x: T) -> impl Future<Output = T::Return> {
    GenFuture(x)
}

/// A wrapper around generators used to implement `Future` for `async`/`await` code.
#[unstable(feature = "gen_future", issue = "50547")]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
struct GenFuture<T: Generator<Yield = ()>>(T);

// We rely on the fact that async/await futures are immovable in order to create
// self-referential borrows in the underlying generator.
impl<T: Generator<Yield = ()>> !Unpin for GenFuture<T> {}

#[unstable(feature = "gen_future", issue = "50547")]
impl<T: Generator<Yield = ()>> Future for GenFuture<T> {
    type Output = T::Return;
    fn poll(self: Pin<&mut Self>, lw: &LocalWaker) -> Poll<Self::Output> {
        set_task_waker(lw, || match unsafe { Pin::get_unchecked_mut(self).0.resume() } {
            GeneratorState::Yielded(()) => Poll::Pending,
            GeneratorState::Complete(x) => Poll::Ready(x),
        })
    }
}

thread_local! {
    static TLS_WAKER: Cell<Option<NonNull<LocalWaker>>> = Cell::new(None);
}

struct SetOnDrop(Option<NonNull<LocalWaker>>);

impl Drop for SetOnDrop {
    fn drop(&mut self) {
        TLS_WAKER.with(|tls_waker| {
            tls_waker.set(self.0.take());
        });
    }
}

#[unstable(feature = "gen_future", issue = "50547")]
/// Sets the thread-local task context used by async/await futures.
pub fn set_task_waker<F, R>(lw: &LocalWaker, f: F) -> R
where
    F: FnOnce() -> R
{
    let old_waker = TLS_WAKER.with(|tls_waker| {
        tls_waker.replace(Some(NonNull::from(lw)))
    });
    let _reset_waker = SetOnDrop(old_waker);
    f()
}

#[unstable(feature = "gen_future", issue = "50547")]
/// Retrieves the thread-local task waker used by async/await futures.
///
/// This function acquires exclusive access to the task waker.
///
/// Panics if no waker has been set or if the waker has already been
/// retrieved by a surrounding call to get_task_waker.
pub fn get_task_waker<F, R>(f: F) -> R
where
    F: FnOnce(&LocalWaker) -> R
{
    let waker_ptr = TLS_WAKER.with(|tls_waker| {
        // Clear the entry so that nested `get_task_waker` calls
        // will fail or set their own value.
        tls_waker.replace(None)
    });
    let _reset_waker = SetOnDrop(waker_ptr);

    let waker_ptr = waker_ptr.expect(
        "TLS LocalWaker not set. This is a rustc bug. \
        Please file an issue on https://github.com/rust-lang/rust.");
    unsafe { f(waker_ptr.as_ref()) }
}

#[unstable(feature = "gen_future", issue = "50547")]
/// Polls a future in the current thread-local task waker.
pub fn poll_with_tls_waker<F>(f: Pin<&mut F>) -> Poll<F::Output>
where
    F: Future
{
    get_task_waker(|lw| F::poll(f, lw))
}
