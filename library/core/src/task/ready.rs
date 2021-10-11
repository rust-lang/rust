use core::convert;
use core::fmt;
use core::ops::{ControlFlow, FromResidual, Try};
use core::task::Poll;

/// Extracts the successful type of a [`Poll<T>`].
///
/// This macro bakes in propagation of [`Pending`] signals by returning early.
///
/// [`Poll<T>`]: crate::task::Poll
/// [`Pending`]: crate::task::Poll::Pending
///
/// # Examples
///
/// ```
/// #![feature(ready_macro)]
///
/// use std::task::{ready, Context, Poll};
/// use std::future::{self, Future};
/// use std::pin::Pin;
///
/// pub fn do_poll(cx: &mut Context<'_>) -> Poll<()> {
///     let mut fut = future::ready(42);
///     let fut = Pin::new(&mut fut);
///
///     let num = ready!(fut.poll(cx));
///     # drop(num);
///     // ... use num
///
///     Poll::Ready(())
/// }
/// ```
///
/// The `ready!` call expands to:
///
/// ```
/// # #![feature(ready_macro)]
/// # use std::task::{Context, Poll};
/// # use std::future::{self, Future};
/// # use std::pin::Pin;
/// #
/// # pub fn do_poll(cx: &mut Context<'_>) -> Poll<()> {
///     # let mut fut = future::ready(42);
///     # let fut = Pin::new(&mut fut);
///     #
/// let num = match fut.poll(cx) {
///     Poll::Ready(t) => t,
///     Poll::Pending => return Poll::Pending,
/// };
///     # drop(num);
///     # // ... use num
///     #
///     # Poll::Ready(())
/// # }
/// ```
#[unstable(feature = "ready_macro", issue = "70922")]
#[rustc_macro_transparency = "semitransparent"]
pub macro ready($e:expr) {
    match $e {
        $crate::task::Poll::Ready(t) => t,
        $crate::task::Poll::Pending => {
            return $crate::task::Poll::Pending;
        }
    }
}

/// Extracts the successful type of a [`Poll<T>`].
///
/// See [`Poll::ready`] for details.
#[unstable(feature = "poll_ready", issue = "89780")]
pub struct Ready<T>(pub(crate) Poll<T>);

#[unstable(feature = "poll_ready", issue = "89780")]
impl<T> Try for Ready<T> {
    type Output = T;
    type Residual = Ready<convert::Infallible>;

    #[inline]
    fn from_output(output: Self::Output) -> Self {
        Ready(Poll::Ready(output))
    }

    #[inline]
    fn branch(self) -> ControlFlow<Self::Residual, Self::Output> {
        match self.0 {
            Poll::Ready(v) => ControlFlow::Continue(v),
            Poll::Pending => ControlFlow::Break(Ready(Poll::Pending)),
        }
    }
}

#[unstable(feature = "poll_ready", issue = "89780")]
impl<T> FromResidual for Ready<T> {
    #[inline]
    fn from_residual(residual: Ready<convert::Infallible>) -> Self {
        match residual.0 {
            Poll::Pending => Ready(Poll::Pending),
        }
    }
}

#[unstable(feature = "poll_ready", issue = "89780")]
impl<T> FromResidual<Ready<convert::Infallible>> for Poll<T> {
    #[inline]
    fn from_residual(residual: Ready<convert::Infallible>) -> Self {
        match residual.0 {
            Poll::Pending => Poll::Pending,
        }
    }
}

#[unstable(feature = "poll_ready", issue = "89780")]
impl<T> fmt::Debug for Ready<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Ready").finish()
    }
}
