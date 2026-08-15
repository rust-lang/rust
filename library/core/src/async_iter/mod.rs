//! Composable asynchronous iteration.
//!
//! If you've found yourself with an asynchronous collection of some kind,
//! and needed to perform an operation on the elements of said collection,
//! you'll quickly run into 'async iterators'. Async Iterators are heavily used in
//! idiomatic asynchronous Rust code, so it's worth becoming familiar with them.
//!
//! Before explaining more, let's talk about how this module is structured:
//!
//! # Organization
//!
//! This module is largely organized by type:
//!
//! * [Traits] are the core portion: these traits define what kind of async iterators
//!   exist and what you can do with them. The methods of these traits are worth
//!   putting some extra study time into.
//! * Functions provide some helpful ways to create some basic async iterators.
//! * Structs are often the return types of the various methods on this
//!   module's traits. You'll usually want to look at the method that creates
//!   the `struct`, rather than the `struct` itself. For more detail about why,
//!   see '[Implementing Async Iterator](#implementing-async-iterator)'.
//!
//! [Traits]: #traits
//!
//! That's it! Let's dig into async iterators.
//!
//! # Async Iterators
//!
//! The heart and soul of this module is the [`AsyncIterator`] trait. The core of
//! [`AsyncIterator`] looks like this:
//!
//! ```
//! # use core::task::{Context, Poll};
//! # use core::pin::Pin;
//! trait AsyncIterator {
//!     type Item;
//!     fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>>;
//! }
//! ```
//!
//! Unlike `Iterator`, `AsyncIterator` makes a distinction between the [`poll_next`]
//! method which is used when implementing an `AsyncIterator`, and a (to-be-implemented)
//! `next` method which is used when consuming an async iterator. Consumers of `AsyncIterator`
//! only need to consider `next`, which when called, returns a future which
//! yields `Option<AsyncIterator::Item>`.
//!
//! The future returned by `next` will yield `Some(Item)` as long as there are
//! elements, and once they've all been exhausted, will yield `None` to indicate
//! that iteration is finished. If we're waiting on something asynchronous to
//! resolve, the future will wait until the async iterator is ready to yield again.
//!
//! Individual async iterators may choose to resume iteration, and so calling `next`
//! again may or may not eventually yield `Some(Item)` again at some point.
//!
//! [`AsyncIterator`]'s full definition includes a number of other methods as well,
//! but they are default methods, built on top of [`poll_next`], and so you get
//! them for free.
//!
//! [`Poll`]: super::task::Poll
//! [`poll_next`]: AsyncIterator::poll_next
//!
//! # Implementing Async Iterator
//!
//! Creating an async iterator of your own involves two steps: creating a `struct` to
//! hold the async iterator's state, and then implementing [`AsyncIterator`] for that
//! `struct`.
//!
//! Let's make an async iterator named `Counter` which counts from `1` to `5`:
//!
//! ```no_run
//! #![feature(async_iterator)]
//! # use core::async_iter::AsyncIterator;
//! # use core::task::{Context, Poll};
//! # use core::pin::Pin;
//!
//! // First, the struct:
//!
//! /// An async iterator which counts from one to five
//! struct Counter {
//!     count: usize,
//! }
//!
//! // we want our count to start at one, so let's add a new() method to help.
//! // This isn't strictly necessary, but is convenient. Note that we start
//! // `count` at zero, we'll see why in `poll_next()`'s implementation below.
//! impl Counter {
//!     fn new() -> Counter {
//!         Counter { count: 0 }
//!     }
//! }
//!
//! // Then, we implement `AsyncIterator` for our `Counter`:
//!
//! impl AsyncIterator for Counter {
//!     // we will be counting with usize
//!     type Item = usize;
//!
//!     // poll_next() is the only required method
//!     fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
//!         // Increment our count. This is why we started at zero.
//!         self.count += 1;
//!
//!         // Check to see if we've finished counting or not.
//!         if self.count < 6 {
//!             Poll::Ready(Some(self.count))
//!         } else {
//!             Poll::Ready(None)
//!         }
//!     }
//! }
//! ```
//!
//! # Laziness
//!
//! Async iterators are *lazy*. This means that just creating an async iterator doesn't
//! _do_ a whole lot. Nothing really happens until you call `poll_next`. This is
//! sometimes a source of confusion when creating an async iterator solely for its side
//! effects. The compiler will warn us about this kind of behavior:
//!
//! ```text
//! warning: unused result that must be used: async iterators do nothing unless polled
//! ```

mod async_iter;
mod from_iter;

pub use async_iter::{AsyncIterator, IntoAsyncIterator};
pub use from_iter::{FromIter, from_iter};
