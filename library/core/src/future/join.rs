#![allow(unused_imports)] // items are used by the macro

use crate::cell::UnsafeCell;
use crate::future::{poll_fn, Future};
use crate::pin::Pin;
use crate::task::Poll;
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
/// let x = join!(one(), two());
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
/// let x = join!(one(), two(), three());
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
        // The future currently being expanded, and the rest
        @rest: ($current:expr, $($rest:tt)*)
    ) => {
        join! {
            @count: ($($count)* _), // Add to the count
            @futures: { $($fut)* $current => ($($count)*), }, // Add the future from @rest with it's position
            @rest: ($($rest)*) // And leave the rest
        }
    },
    // Now generate the output future
    (
        @count: ($($count:tt)*),
        @futures: {
            $( $fut:expr => ( $($pos:tt)* ), )*
        },
        @rest: ()
    ) => {{
        let mut futures = ( $( MaybeDone::Future($fut), )* );

        poll_fn(move |cx| {
            let mut done = true;

            $(
                // Extract the future from the tuple
                let ( $($pos,)* fut, .. ) = &mut futures;

                // SAFETY: the futures are never moved
                done &= unsafe { Pin::new_unchecked(fut).poll(cx).is_ready() };
            )*

            if done {
                Poll::Ready(($({
                    let ( $($pos,)* fut, .. ) = &mut futures;

                    // SAFETY: the futures are never moved
                    unsafe { Pin::new_unchecked(fut).take_output().unwrap() }
                }),*))
            } else {
                Poll::Pending
            }
        }).await
    }}
}

/// Future used by `join!` that stores it's output to
/// be later taken and doesn't panic when polled after ready.
#[allow(dead_code)]
#[unstable(feature = "future_join", issue = "none")]
enum MaybeDone<F: Future> {
    Future(F),
    Done(F::Output),
    Took,
}

#[unstable(feature = "future_join", issue = "none")]
impl<F: Future + Unpin> Unpin for MaybeDone<F> {}

#[unstable(feature = "future_join", issue = "none")]
impl<F: Future> MaybeDone<F> {
    #[allow(dead_code)]
    fn take_output(self: Pin<&mut Self>) -> Option<F::Output> {
        unsafe {
            match &*self {
                MaybeDone::Done(_) => match mem::replace(self.get_unchecked_mut(), Self::Took) {
                    MaybeDone::Done(val) => Some(val),
                    _ => unreachable!(),
                },
                _ => None,
            }
        }
    }
}

#[unstable(feature = "future_join", issue = "none")]
impl<F: Future> Future for MaybeDone<F> {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
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
