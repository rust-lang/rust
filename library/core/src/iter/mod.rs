//! Composable external iteration.
//!
//! If you've found yourself with a collection of some kind, and needed to
//! perform an operation on the elements of said collection, you'll quickly run
//! into 'iterators'. Iterators are heavily used in idiomatic Rust code, so
//! it's worth becoming familiar with them.
//!
//! Before explaining more, let's talk about how this module is structured:
//!
//! # Organization
//!
//! This module is largely organized by type:
//!
//! * [Traits] are the core portion: these traits define what kind of iterators
//!   exist and what you can do with them. The methods of these traits are worth
//!   putting some extra study time into.
//! * [Functions] provide some helpful ways to create some basic iterators.
//! * [Structs] are often the return types of the various methods on this
//!   module's traits. You'll usually want to look at the method that creates
//!   the `struct`, rather than the `struct` itself. For more detail about why,
//!   see '[Implementing Iterator](#implementing-iterator)'.
//!
//! [Traits]: #traits
//! [Functions]: #functions
//! [Structs]: #structs
//!
//! That's it! Let's dig into iterators.
//!
//! # Iterator
//!
//! The heart and soul of this module is the [`Iterator`] trait. The core of
//! [`Iterator`] looks like this:
//!
//! ```
//! trait Iterator {
//!     type Item;
//!     fn next(&mut self) -> Option<Self::Item>;
//! }
//! ```
//!
//! An iterator has a method, [`next`], which when called, returns an
//! <code>[Option]\<Item></code>. Calling [`next`] will return [`Some(Item)`] as long as there
//! are elements, and once they've all been exhausted, will return `None` to
//! indicate that iteration is finished. Individual iterators may choose to
//! resume iteration, and so calling [`next`] again may or may not eventually
//! start returning [`Some(Item)`] again at some point (for example, see [`TryIter`]).
//!
//! [`Iterator`]'s full definition includes a number of other methods as well,
//! but they are default methods, built on top of [`next`], and so you get
//! them for free.
//!
//! Iterators are also composable, and it's common to chain them together to do
//! more complex forms of processing. See the [Adapters](#adapters) section
//! below for more details.
//!
//! [`Some(Item)`]: Some
//! [`next`]: Iterator::next
//! [`TryIter`]: ../../std/sync/mpsc/struct.TryIter.html
//!
//! # The three forms of iteration
//!
//! There are three common methods which can create iterators from a collection:
//!
//! * `iter()`, which iterates over `&T`.
//! * `iter_mut()`, which iterates over `&mut T`.
//! * `into_iter()`, which iterates over `T`.
//!
//! Various things in the standard library may implement one or more of the
//! three, where appropriate.
//!
//! # Implementing Iterator
//!
//! Creating an iterator of your own involves two steps: creating a `struct` to
//! hold the iterator's state, and then implementing [`Iterator`] for that `struct`.
//! This is why there are so many `struct`s in this module: there is one for
//! each iterator and iterator adapter.
//!
//! Let's make an iterator named `Counter` which counts from `1` to `5`:
//!
//! ```
//! // First, the struct:
//!
//! /// An iterator which counts from one to five
//! struct Counter {
//!     count: usize,
//! }
//!
//! // we want our count to start at one, so let's add a new() method to help.
//! // This isn't strictly necessary, but is convenient. Note that we start
//! // `count` at zero, we'll see why in `next()`'s implementation below.
//! impl Counter {
//!     fn new() -> Counter {
//!         Counter { count: 0 }
//!     }
//! }
//!
//! // Then, we implement `Iterator` for our `Counter`:
//!
//! impl Iterator for Counter {
//!     // we will be counting with usize
//!     type Item = usize;
//!
//!     // next() is the only required method
//!     fn next(&mut self) -> Option<Self::Item> {
//!         // Increment our count. This is why we started at zero.
//!         self.count += 1;
//!
//!         // Check to see if we've finished counting or not.
//!         if self.count < 6 {
//!             Some(self.count)
//!         } else {
//!             None
//!         }
//!     }
//! }
//!
//! // And now we can use it!
//!
//! let mut counter = Counter::new();
//!
//! assert_eq!(counter.next(), Some(1));
//! assert_eq!(counter.next(), Some(2));
//! assert_eq!(counter.next(), Some(3));
//! assert_eq!(counter.next(), Some(4));
//! assert_eq!(counter.next(), Some(5));
//! assert_eq!(counter.next(), None);
//! ```
//!
//! Calling [`next`] this way gets repetitive. Rust has a construct which can
//! call [`next`] on your iterator, until it reaches `None`. Let's go over that
//! next.
//!
//! Also note that `Iterator` provides a default implementation of methods such as `nth` and `fold`
//! which call `next` internally. However, it is also possible to write a custom implementation of
//! methods like `nth` and `fold` if an iterator can compute them more efficiently without calling
//! `next`.
//!
//! # `for` loops and `IntoIterator`
//!
//! Rust's `for` loop syntax is actually sugar for iterators. Here's a basic
//! example of `for`:
//!
//! ```
//! let values = vec![1, 2, 3, 4, 5];
//!
//! for x in values {
//!     println!("{}", x);
//! }
//! ```
//!
//! This will print the numbers one through five, each on their own line. But
//! you'll notice something here: we never called anything on our vector to
//! produce an iterator. What gives?
//!
//! There's a trait in the standard library for converting something into an
//! iterator: [`IntoIterator`]. This trait has one method, [`into_iter`],
//! which converts the thing implementing [`IntoIterator`] into an iterator.
//! Let's take a look at that `for` loop again, and what the compiler converts
//! it into:
//!
//! [`into_iter`]: IntoIterator::into_iter
//!
//! ```
//! let values = vec![1, 2, 3, 4, 5];
//!
//! for x in values {
//!     println!("{}", x);
//! }
//! ```
//!
//! Rust de-sugars this into:
//!
//! ```
//! let values = vec![1, 2, 3, 4, 5];
//! {
//!     let result = match IntoIterator::into_iter(values) {
//!         mut iter => loop {
//!             let next;
//!             match iter.next() {
//!                 Some(val) => next = val,
//!                 None => break,
//!             };
//!             let x = next;
//!             let () = { println!("{}", x); };
//!         },
//!     };
//!     result
//! }
//! ```
//!
//! First, we call `into_iter()` on the value. Then, we match on the iterator
//! that returns, calling [`next`] over and over until we see a `None`. At
//! that point, we `break` out of the loop, and we're done iterating.
//!
//! There's one more subtle bit here: the standard library contains an
//! interesting implementation of [`IntoIterator`]:
//!
//! ```ignore (only-for-syntax-highlight)
//! impl<I: Iterator> IntoIterator for I
//! ```
//!
//! In other words, all [`Iterator`]s implement [`IntoIterator`], by just
//! returning themselves. This means two things:
//!
//! 1. If you're writing an [`Iterator`], you can use it with a `for` loop.
//! 2. If you're creating a collection, implementing [`IntoIterator`] for it
//!    will allow your collection to be used with the `for` loop.
//!
//! # Iterating by reference
//!
//! Since [`into_iter()`] takes `self` by value, using a `for` loop to iterate
//! over a collection consumes that collection. Often, you may want to iterate
//! over a collection without consuming it. Many collections offer methods that
//! provide iterators over references, conventionally called `iter()` and
//! `iter_mut()` respectively:
//!
//! ```
//! let mut values = vec![41];
//! for x in values.iter_mut() {
//!     *x += 1;
//! }
//! for x in values.iter() {
//!     assert_eq!(*x, 42);
//! }
//! assert_eq!(values.len(), 1); // `values` is still owned by this function.
//! ```
//!
//! If a collection type `C` provides `iter()`, it usually also implements
//! `IntoIterator` for `&C`, with an implementation that just calls `iter()`.
//! Likewise, a collection `C` that provides `iter_mut()` generally implements
//! `IntoIterator` for `&mut C` by delegating to `iter_mut()`. This enables a
//! convenient shorthand:
//!
//! ```
//! let mut values = vec![41];
//! for x in &mut values { // same as `values.iter_mut()`
//!     *x += 1;
//! }
//! for x in &values { // same as `values.iter()`
//!     assert_eq!(*x, 42);
//! }
//! assert_eq!(values.len(), 1);
//! ```
//!
//! While many collections offer `iter()`, not all offer `iter_mut()`. For
//! example, mutating the keys of a [`HashSet<T>`] or [`HashMap<K, V>`] could
//! put the collection into an inconsistent state if the key hashes change, so
//! these collections only offer `iter()`.
//!
//! [`into_iter()`]: IntoIterator::into_iter
//! [`HashSet<T>`]: ../../std/collections/struct.HashSet.html
//! [`HashMap<K, V>`]: ../../std/collections/struct.HashMap.html
//!
//! # Adapters
//!
//! Functions which take an [`Iterator`] and return another [`Iterator`] are
//! often called 'iterator adapters', as they're a form of the 'adapter
//! pattern'.
//!
//! Common iterator adapters include [`map`], [`take`], and [`filter`].
//! For more, see their documentation.
//!
//! If an iterator adapter panics, the iterator will be in an unspecified (but
//! memory safe) state.  This state is also not guaranteed to stay the same
//! across versions of Rust, so you should avoid relying on the exact values
//! returned by an iterator which panicked.
//!
//! [`map`]: Iterator::map
//! [`take`]: Iterator::take
//! [`filter`]: Iterator::filter
//!
//! # Laziness
//!
//! Iterators (and iterator [adapters](#adapters)) are *lazy*. This means that
//! just creating an iterator doesn't _do_ a whole lot. Nothing really happens
//! until you call [`next`]. This is sometimes a source of confusion when
//! creating an iterator solely for its side effects. For example, the [`map`]
//! method calls a closure on each element it iterates over:
//!
//! ```
//! # #![allow(unused_must_use)]
//! let v = vec![1, 2, 3, 4, 5];
//! v.iter().map(|x| println!("{}", x));
//! ```
//!
//! This will not print any values, as we only created an iterator, rather than
//! using it. The compiler will warn us about this kind of behavior:
//!
//! ```text
//! warning: unused result that must be used: iterators are lazy and
//! do nothing unless consumed
//! ```
//!
//! The idiomatic way to write a [`map`] for its side effects is to use a
//! `for` loop or call the [`for_each`] method:
//!
//! ```
//! let v = vec![1, 2, 3, 4, 5];
//!
//! v.iter().for_each(|x| println!("{}", x));
//! // or
//! for x in &v {
//!     println!("{}", x);
//! }
//! ```
//!
//! [`map`]: Iterator::map
//! [`for_each`]: Iterator::for_each
//!
//! Another common way to evaluate an iterator is to use the [`collect`]
//! method to produce a new collection.
//!
//! [`collect`]: Iterator::collect
//!
//! # Infinity
//!
//! Iterators do not have to be finite. As an example, an open-ended range is
//! an infinite iterator:
//!
//! ```
//! let numbers = 0..;
//! ```
//!
//! It is common to use the [`take`] iterator adapter to turn an infinite
//! iterator into a finite one:
//!
//! ```
//! let numbers = 0..;
//! let five_numbers = numbers.take(5);
//!
//! for number in five_numbers {
//!     println!("{}", number);
//! }
//! ```
//!
//! This will print the numbers `0` through `4`, each on their own line.
//!
//! Bear in mind that methods on infinite iterators, even those for which a
//! result can be determined mathematically in finite time, might not terminate.
//! Specifically, methods such as [`min`], which in the general case require
//! traversing every element in the iterator, are likely not to return
//! successfully for any infinite iterators.
//!
//! ```no_run
//! let ones = std::iter::repeat(1);
//! let least = ones.min().unwrap(); // Oh no! An infinite loop!
//! // `ones.min()` causes an infinite loop, so we won't reach this point!
//! println!("The smallest number one is {}.", least);
//! ```
//!
//! [`take`]: Iterator::take
//! [`min`]: Iterator::min

#![stable(feature = "rust1", since = "1.0.0")]

#[stable(feature = "rust1", since = "1.0.0")]
pub use self::traits::Iterator;

#[unstable(
    feature = "step_trait",
    reason = "likely to be replaced by finer-grained traits",
    issue = "42168"
)]
pub use self::range::Step;

#[stable(feature = "iter_empty", since = "1.2.0")]
pub use self::sources::{empty, Empty};
#[stable(feature = "iter_from_fn", since = "1.34.0")]
pub use self::sources::{from_fn, FromFn};
#[stable(feature = "iter_once", since = "1.2.0")]
pub use self::sources::{once, Once};
#[stable(feature = "iter_once_with", since = "1.43.0")]
pub use self::sources::{once_with, OnceWith};
#[stable(feature = "rust1", since = "1.0.0")]
pub use self::sources::{repeat, Repeat};
#[stable(feature = "iterator_repeat_with", since = "1.28.0")]
pub use self::sources::{repeat_with, RepeatWith};
#[stable(feature = "iter_successors", since = "1.34.0")]
pub use self::sources::{successors, Successors};

#[stable(feature = "fused", since = "1.26.0")]
pub use self::traits::FusedIterator;
#[unstable(issue = "none", feature = "inplace_iteration")]
pub use self::traits::InPlaceIterable;
#[unstable(feature = "trusted_len", issue = "37572")]
pub use self::traits::TrustedLen;
#[unstable(feature = "trusted_step", issue = "85731")]
pub use self::traits::TrustedStep;
#[stable(feature = "rust1", since = "1.0.0")]
pub use self::traits::{
    DoubleEndedIterator, ExactSizeIterator, Extend, FromIterator, IntoIterator, Product, Sum,
};

#[unstable(feature = "iter_zip", issue = "83574")]
pub use self::adapters::zip;
#[stable(feature = "iter_cloned", since = "1.1.0")]
pub use self::adapters::Cloned;
#[stable(feature = "iter_copied", since = "1.36.0")]
pub use self::adapters::Copied;
#[stable(feature = "iterator_flatten", since = "1.29.0")]
pub use self::adapters::Flatten;
#[stable(feature = "iter_map_while", since = "1.57.0")]
pub use self::adapters::MapWhile;
#[unstable(feature = "inplace_iteration", issue = "none")]
pub use self::adapters::SourceIter;
#[stable(feature = "iterator_step_by", since = "1.28.0")]
pub use self::adapters::StepBy;
#[unstable(feature = "trusted_random_access", issue = "none")]
pub use self::adapters::TrustedRandomAccess;
#[unstable(feature = "trusted_random_access", issue = "none")]
pub use self::adapters::TrustedRandomAccessNoCoerce;
#[stable(feature = "rust1", since = "1.0.0")]
pub use self::adapters::{
    Chain, Cycle, Enumerate, Filter, FilterMap, FlatMap, Fuse, Inspect, Map, Peekable, Rev, Scan,
    Skip, SkipWhile, Take, TakeWhile, Zip,
};
#[stable(feature = "iter_intersperse", since = "1.56.0")]
pub use self::adapters::{Intersperse, IntersperseWith};

pub(crate) use self::adapters::process_results;

mod adapters;
mod range;
mod sources;
mod traits;
