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

mod future;
mod into_future;
mod join;
mod pending;
mod poll_fn;
mod ready;

#[stable(feature = "futures_api", since = "1.36.0")]
pub use self::future::Future;

#[unstable(feature = "future_join", issue = "91642")]
pub use self::join::join;

#[stable(feature = "into_future", since = "1.64.0")]
pub use into_future::IntoFuture;

#[stable(feature = "future_readiness_fns", since = "1.48.0")]
pub use pending::{pending, Pending};
#[stable(feature = "future_readiness_fns", since = "1.48.0")]
pub use ready::{ready, Ready};

#[stable(feature = "future_poll_fn", since = "1.64.0")]
pub use poll_fn::{poll_fn, PollFn};

/// This type is needed because:
///
/// a) Generators cannot implement `for<'a, 'b> Generator<&'a mut Context<'b>>`, so we need to pass
///    a raw pointer (see <https://github.com/rust-lang/rust/issues/68923>).
/// b) Raw pointers and `NonNull` aren't `Send` or `Sync`, so that would make every single future
///    non-Send/Sync as well, and we don't want that.
///
/// It also simplifies the HIR lowering of `.await`.
#[lang = "ResumeTy"]
#[doc(hidden)]
#[unstable(feature = "gen_future", issue = "50547")]
#[derive(Debug, Copy, Clone)]
pub struct ResumeTy(NonNull<Context<'static>>);

#[unstable(feature = "gen_future", issue = "50547")]
unsafe impl Send for ResumeTy {}

#[unstable(feature = "gen_future", issue = "50547")]
unsafe impl Sync for ResumeTy {}

#[lang = "get_context"]
#[doc(hidden)]
#[unstable(feature = "gen_future", issue = "50547")]
#[must_use]
#[inline]
pub unsafe fn get_context<'a, 'b>(cx: ResumeTy) -> &'a mut Context<'b> {
    // SAFETY: the caller must guarantee that `cx.0` is a valid pointer
    // that fulfills all the requirements for a mutable reference.
    unsafe { &mut *cx.0.as_ptr().cast() }
}

// FIXME(swatinem): This fn is currently needed to work around shortcomings
// in type and lifetime inference.
// See the comment at the bottom of `LoweringContext::make_async_expr` and
// <https://github.com/rust-lang/rust/issues/104826>.
#[doc(hidden)]
#[unstable(feature = "gen_future", issue = "50547")]
#[inline]
#[lang = "identity_future"]
pub const fn identity_future<O, Fut: Future<Output = O>>(f: Fut) -> Fut {
    f
}
