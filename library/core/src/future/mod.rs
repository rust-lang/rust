#![stable(feature = "futures_api", since = "1.36.0")]

//! Asynchronous basic functionality.
//!
//! Please see the fundamental [`async`] and [`await`] keywords and the [async book]
//! for more information on asynchronous programming in Rust.
//!
//! [`async`]: ../../std/keyword.async.html
//! [`await`]: ../../std/keyword.await.html
//! [async book]: https://rust-lang.github.io/async-book/

use crate::ptr::NonNull;
use crate::task::Context;

mod async_drop;
mod future;
mod into_future;
mod join;
mod pending;
mod poll_fn;
mod ready;

#[unstable(feature = "async_drop", issue = "126482")]
pub use async_drop::{AsyncDrop, async_drop_in_place};
#[stable(feature = "into_future", since = "1.64.0")]
pub use into_future::IntoFuture;
#[stable(feature = "future_readiness_fns", since = "1.48.0")]
pub use pending::{Pending, pending};
#[stable(feature = "future_poll_fn", since = "1.64.0")]
pub use poll_fn::{PollFn, poll_fn};
#[stable(feature = "future_readiness_fns", since = "1.48.0")]
pub use ready::{Ready, ready};

#[stable(feature = "futures_api", since = "1.36.0")]
pub use self::future::Future;
#[unstable(feature = "future_join", issue = "91642")]
pub use self::join::join;

/// This type is needed because:
///
/// a) Coroutines cannot implement `for<'a, 'b> Coroutine<&'a mut Context<'b>>`, so we need to pass
///    a raw pointer (see <https://github.com/rust-lang/rust/issues/68923>).
/// b) Raw pointers and `NonNull` aren't `Send` or `Sync`, so that would make every single future
///    non-Send/Sync as well, and we don't want that.
///
/// It also simplifies the HIR lowering of `.await`.
#[lang = "ResumeTy"]
#[doc(hidden)]
#[unstable(feature = "gen_future", issue = "none")]
#[derive(Debug, Copy, Clone)]
pub struct ResumeTy(NonNull<Context<'static>>);

#[unstable(feature = "gen_future", issue = "none")]
unsafe impl Send for ResumeTy {}

#[unstable(feature = "gen_future", issue = "none")]
unsafe impl Sync for ResumeTy {}

/// Helper function to wrap a `&mut Context` into a `ResumeTy` that can be passed to a coroutine.
/// This is used by `CoroutineFuture` type.
#[must_use]
#[inline]
pub(crate) unsafe fn wrap_context(cx: &mut Context<'_>) -> ResumeTy {
    // SAFETY: `cx` is a valid mutable reference, its pointer is non-null.
    // The cast to `'static` is sound because the `ResumeTy`
    // is only unwrapped by `get_context`.
    ResumeTy(unsafe { NonNull::new_unchecked((&raw mut *cx).cast()) })
}

#[lang = "get_context"]
#[doc(hidden)]
#[unstable(feature = "gen_future", issue = "none")]
#[must_use]
#[inline]
pub unsafe fn get_context<'a, 'b>(cx: ResumeTy) -> &'a mut Context<'b> {
    // SAFETY: the caller must guarantee that `cx.0` is a valid pointer
    // that fulfills all the requirements for a mutable reference.
    unsafe { &mut *cx.0.as_ptr().cast() }
}
