#![allow(unused_imports, unused_macros)] // items are used by the macro

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
pub macro join( $($fut:expr),+ $(,)? ) {
    // Funnel through an internal macro not to leak implementation details.
    join_internal! {
        current_position: []
        futures_and_positions: []
        munching: [ $($fut)+ ]
    }
}

// FIXME(danielhenrymantilla): a private macro should need no stability guarantee.
#[unstable(feature = "future_join", issue = "91642")]
/// To be able to *name* the i-th future in the tuple (say we want the .4-th),
/// the following trick will be used: `let (_, _, _, _, it, ..) = tuple;`
/// In order to do that, we need to generate a `i`-long repetition of `_`,
/// for each i-th fut. Hence the recursive muncher approach.
macro join_internal {
    // Recursion step: map each future with its "position" (underscore count).
    (
        // Accumulate a token for each future that has been expanded: "_ _ _".
        current_position: [
            $($underscores:tt)*
        ]
        // Accumulate Futures and their positions in the tuple: `_0th ()   _1st ( _ ) …`.
        futures_and_positions: [
            $($acc:tt)*
        ]
        // Munch one future.
        munching: [
            $current:tt
            $($rest:tt)*
        ]
    ) => (
        join_internal! {
            current_position: [
                $($underscores)*
                _
            ]
            futures_and_positions: [
                $($acc)*
                $current ( $($underscores)* )
            ]
            munching: [
                $($rest)*
            ]
        }
    ),

    // End of recursion: generate the output future.
    (
        current_position: $_:tt
        futures_and_positions: [
            $(
                $fut_expr:tt ( $($pos:tt)* )
            )*
        ]
        // Nothing left to munch.
        munching: []
    ) => (
        match ( $( MaybeDone::Future($fut_expr), )* ) { futures => async {
            let mut futures = futures;
            // SAFETY: this is `pin_mut!`.
            let mut futures = unsafe { Pin::new_unchecked(&mut futures) };
            poll_fn(move |cx| {
                let mut done = true;
                // For each `fut`, pin-project to it, and poll it.
                $(
                    // SAFETY: pinning projection
                    let fut = unsafe {
                        futures.as_mut().map_unchecked_mut(|it| {
                            let ( $($pos,)* fut, .. ) = it;
                            fut
                        })
                    };
                    // Despite how tempting it may be to `let () = fut.poll(cx).ready()?;`
                    // doing so would defeat the point of `join!`: to start polling eagerly all
                    // of the futures, to allow parallelizing the waits.
                    done &= fut.poll(cx).is_ready();
                )*
                if !done {
                    return Poll::Pending;
                }
                // All ready; time to extract all the outputs.

                // SAFETY: `.take_output()` does not break the `Pin` invariants for that `fut`.
                let futures = unsafe {
                    futures.as_mut().get_unchecked_mut()
                };
                Poll::Ready(
                    ($(
                        {
                            let ( $($pos,)* fut, .. ) = &mut *futures;
                            fut.take_output().unwrap()
                        }
                    ),*) // <- no trailing comma since we don't want 1-tuples.
                )
            }).await
        }}
    ),
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
    Taken,
}

#[unstable(feature = "future_join", issue = "91642")]
impl<F: Future> MaybeDone<F> {
    pub fn take_output(&mut self) -> Option<F::Output> {
        match *self {
            MaybeDone::Done(_) => match mem::replace(self, Self::Taken) {
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
            // Do not mix match ergonomics with unsafe.
            match *self.as_mut().get_unchecked_mut() {
                MaybeDone::Future(ref mut f) => {
                    let val = Pin::new_unchecked(f).poll(cx).ready()?;
                    self.set(Self::Done(val));
                }
                MaybeDone::Done(_) => {}
                MaybeDone::Taken => unreachable!(),
            }
        }

        Poll::Ready(())
    }
}
