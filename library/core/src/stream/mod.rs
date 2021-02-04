//! Composable asynchronous iteration.
//!
//! If futures are asynchronous values, then streams are asynchronous
//! iterators. If you've found yourself with an asynchronous collection of some kind,
//! and needed to perform an operation on the elements of said collection,
//! you'll quickly run into 'streams'. Streams are heavily used in idiomatic
//! asynchronous Rust code, so it's worth becoming familiar with them.
//!
//! Before explaining more, let's talk about how this module is structured:
//!
//! # Organization
//!
//! This module is largely organized by type:
//!
//! * [Traits] are the core portion: these traits define what kind of streams
//!   exist and what you can do with them. The methods of these traits are worth
//!   putting some extra study time into.
//! * Functions provide some helpful ways to create some basic streams.
//! * Structs are often the return types of the various methods on this
//!   module's traits. You'll usually want to look at the method that creates
//!   the `struct`, rather than the `struct` itself. For more detail about why,
//!   see '[Implementing Stream](#implementing-stream)'.
//!
//! [Traits]: #traits
//!
//! That's it! Let's dig into streams.
//!
//! # Stream
//!
//! The heart and soul of this module is the [`Stream`] trait. The core of
//! [`Stream`] looks like this:
//!
//! ```
//! # use core::task::{Context, Poll};
//! # use core::pin::Pin;
//! trait Stream {
//!     type Item;
//!     fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>>;
//! }
//! ```
//!
//! Unlike `Iterator`, `Stream` makes a distinction between the [`poll_next`]
//! method which is used when implementing a `Stream`, and a (to-be-implemented)
//! `next` method which is used when consuming a stream. Consumers of `Stream`
//! only need to consider `next`, which when called, returns a future which
//! yields `Option<Stream::Item>`.
//!
//! The future returned by `next` will yield `Some(Item)` as long as there are
//! elements, and once they've all been exhausted, will yield `None` to indicate
//! that iteration is finished. If we're waiting on something asynchronous to
//! resolve, the future will wait until the stream is ready to yield again.
//!
//! Individual streams may choose to resume iteration, and so calling `next`
//! again may or may not eventually yield `Some(Item)` again at some point.
//!
//! [`Stream`]'s full definition includes a number of other methods as well,
//! but they are default methods, built on top of [`poll_next`], and so you get
//! them for free.
//!
//! [`Poll`]: super::task::Poll
//! [`poll_next`]: Stream::poll_next
//!
//! # Implementing Stream
//!
//! Creating a stream of your own involves two steps: creating a `struct` to
//! hold the stream's state, and then implementing [`Stream`] for that
//! `struct`.
//!
//! Let's make a stream named `Counter` which counts from `1` to `5`:
//!
//! ```no_run
//! #![feature(async_stream)]
//! # use core::stream::Stream;
//! # use core::task::{Context, Poll};
//! # use core::pin::Pin;
//!
//! // First, the struct:
//!
//! /// A stream which counts from one to five
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
//! // Then, we implement `Stream` for our `Counter`:
//!
//! impl Stream for Counter {
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
//! Streams are *lazy*. This means that just creating a stream doesn't _do_ a
//! whole lot. Nothing really happens until you call `next`. This is sometimes a
//! source of confusion when creating a stream solely for its side effects. The
//! compiler will warn us about this kind of behavior:
//!
//! ```text
//! warning: unused result that must be used: streams do nothing unless polled
//! ```

mod from_iter;
mod stream;

pub use from_iter::{from_iter, FromIter};
pub use stream::Stream;
