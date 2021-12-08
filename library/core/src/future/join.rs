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
    ) => {{
        async move {
            // The futures and whether they have completed
            let mut state = ( $( UnsafeCell::new(($fut, false)), )* );

            // Make sure the futures don't panic
            // if polled after completion, and
            // store their output separately
            let mut futures = ($(
                ({
                    let ( $($pos,)* state, .. ) = &state;

                    poll_fn(move |cx| {
                        // SAFETY: each future borrows a distinct element
                        // of the tuple
                        let (fut, done) = unsafe { &mut *state.get() };

                        if *done {
                            return Poll::Ready(None)
                        }

                        // SAFETY: The futures are never moved
                        match unsafe { Pin::new_unchecked(fut).poll(cx) } {
                            Poll::Ready(val) => {
                                *done = true;
                                Poll::Ready(Some(val))
                            }
                            Poll::Pending => Poll::Pending
                        }
                    })
                }, None),
            )*);

            poll_fn(move |cx| {
                let mut done = true;

                $(
                    let ( $($pos,)* (fut, out), .. ) = &mut futures;

                    // SAFETY: The futures are never moved
                    match unsafe { Pin::new_unchecked(fut).poll(cx) } {
                        Poll::Ready(Some(val)) => *out = Some(val),
                        // the future was already done
                        Poll::Ready(None) => {},
                        Poll::Pending => done = false,
                    }
                )*

                if done {
                    // Extract all the outputs
                    Poll::Ready(($({
                        let ( $($pos,)* (_, val), .. ) = &mut futures;
                        val.unwrap()
                    }),*))
                } else {
                    Poll::Pending
                }
            }).await
        }
    }}
}
