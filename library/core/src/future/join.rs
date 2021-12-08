#![allow(unused_imports)] // items are used by the macro

use crate::cell::UnsafeCell;
use crate::future::{poll_fn, Future};
use crate::mem;
use crate::pin::Pin;
use crate::task::{Context, Poll};

/// Polls multiple futures simultaneously, returning a tuple
/// of all results once complete.
///
/// While `join!(a, b)` is similar to `(a.await, b.await)`,
/// `join!` polls both futures concurrently and is therefore more efficient.
///
/// # Examples
///
/// ```
/// #![feature(future_join, future_poll_fn)]
///
/// use std::future::join;
///
/// async fn one() -> usize { 1 }
/// async fn two() -> usize { 2 }
///
/// # let _ =  async {
/// let x = join!(one(), two()).await;
/// assert_eq!(x, (1, 2));
/// # };
/// ```
///
/// `join!` is variadic, so you can pass any number of futures:
///
/// ```
/// #![feature(future_join, future_poll_fn)]
///
/// use std::future::join;
///
/// async fn one() -> usize { 1 }
/// async fn two() -> usize { 2 }
/// async fn three() -> usize { 3 }
///
/// # let _ = async {
/// let x = join!(one(), two(), three()).await;
/// assert_eq!(x, (1, 2, 3));
/// # };
/// ```
#[unstable(feature = "future_join", issue = "91642")]
pub macro join {
    ( $($fut:expr),* $(,)?) => {
        join! { @count: (), @futures: {}, @rest: ($($fut,)*) }
    },
    // Recurse until we have the position of each future in the tuple
    (
        // A token for each future that has been expanded: "_ _ _"
        @count: ($($count:tt)*),
        // Futures and their positions in the tuple: "{ a => (_), b => (_ _)) }"
        @futures: { $($fut:tt)* },
        // Take a future from @rest to expand
        @rest: ($current:expr, $($rest:tt)*)
    ) => {
        join! {
            @count: ($($count)* _),
            @futures: { $($fut)* $current => ($($count)*), },
            @rest: ($($rest)*)
        }
    },
    // Now generate the output future
    (
        @count: ($($count:tt)*),
        @futures: {
            $( $(@$f:tt)? $fut:expr => ( $($pos:tt)* ), )*
        },
        @rest: ()
    ) => {
        async move {
            let mut futures = ( $( MaybeDone::Future($fut), )* );

            poll_fn(move |cx| {
                let mut done = true;

                $(
                    let ( $($pos,)* fut, .. ) = &mut futures;

                    // SAFETY: The futures are never moved
                    done &= unsafe { Pin::new_unchecked(fut).poll(cx).is_ready() };
                )*

                if done {
                    // Extract all the outputs
                    Poll::Ready(($({
                        let ( $($pos,)* fut, .. ) = &mut futures;

                        fut.take_output().unwrap()
                    }),*))
                } else {
                    Poll::Pending
                }
            }).await
        }
    }
}

/// Future used by `join!` that stores it's output to
/// be later taken and doesn't panic when polled after ready.
///
/// This type is public in a private module for use by the macro.
#[allow(missing_debug_implementations)]
#[unstable(feature = "future_join", issue = "91642")]
pub enum MaybeDone<F: Future> {
    Future(F),
    Done(F::Output),
    Took,
}

#[unstable(feature = "future_join", issue = "91642")]
impl<F: Future> MaybeDone<F> {
    pub fn take_output(&mut self) -> Option<F::Output> {
        match &*self {
            MaybeDone::Done(_) => match mem::replace(self, Self::Took) {
                MaybeDone::Done(val) => Some(val),
                _ => unreachable!(),
            },
            _ => None,
        }
    }
}

#[unstable(feature = "future_join", issue = "91642")]
impl<F: Future> Future for MaybeDone<F> {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // SAFETY: pinning in structural for `f`
        unsafe {
            match self.as_mut().get_unchecked_mut() {
                MaybeDone::Future(f) => match Pin::new_unchecked(f).poll(cx) {
                    Poll::Ready(val) => self.set(Self::Done(val)),
                    Poll::Pending => return Poll::Pending,
                },
                MaybeDone::Done(_) => {}
                MaybeDone::Took => unreachable!(),
            }
        }

        Poll::Ready(())
    }
}
