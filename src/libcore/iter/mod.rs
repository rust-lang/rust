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
//! [`Option`]`<Item>`. [`next`] will return `Some(Item)` as long as there
//! are elements, and once they've all been exhausted, will return `None` to
//! indicate that iteration is finished. Individual iterators may choose to
//! resume iteration, and so calling [`next`] again may or may not eventually
//! start returning `Some(Item)` again at some point.
//!
//! [`Iterator`]'s full definition includes a number of other methods as well,
//! but they are default methods, built on top of [`next`], and so you get
//! them for free.
//!
//! Iterators are also composable, and it's common to chain them together to do
//! more complex forms of processing. See the [Adapters](#adapters) section
//! below for more details.
//!
//! [`Iterator`]: trait.Iterator.html
//! [`next`]: trait.Iterator.html#tymethod.next
//! [`Option`]: ../../std/option/enum.Option.html
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
//! hold the iterator's state, and then `impl`ementing [`Iterator`] for that
//! `struct`. This is why there are so many `struct`s in this module: there is
//! one for each iterator and iterator adapter.
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
//! let x = counter.next().unwrap();
//! println!("{}", x);
//!
//! let x = counter.next().unwrap();
//! println!("{}", x);
//!
//! let x = counter.next().unwrap();
//! println!("{}", x);
//!
//! let x = counter.next().unwrap();
//! println!("{}", x);
//!
//! let x = counter.next().unwrap();
//! println!("{}", x);
//! ```
//!
//! This will print `1` through `5`, each on their own line.
//!
//! Calling `next()` this way gets repetitive. Rust has a construct which can
//! call `next()` on your iterator, until it reaches `None`. Let's go over that
//! next.
//!
//! Also note that `Iterator` provides a default implementation of methods such as `nth` and `fold`
//! which call `next` internally. However, it is also possible to write a custom implementation of
//! methods like `nth` and `fold` if an iterator can compute them more efficiently without calling
//! `next`.
//!
//! # for Loops and IntoIterator
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
//! [`IntoIterator`]: trait.IntoIterator.html
//! [`into_iter`]: trait.IntoIterator.html#tymethod.into_iter
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
//! # Adapters
//!
//! Functions which take an [`Iterator`] and return another [`Iterator`] are
//! often called 'iterator adapters', as they're a form of the 'adapter
//! pattern'.
//!
//! Common iterator adapters include [`map`], [`take`], and [`filter`].
//! For more, see their documentation.
//!
//! [`map`]: trait.Iterator.html#method.map
//! [`take`]: trait.Iterator.html#method.take
//! [`filter`]: trait.Iterator.html#method.filter
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
//! `for` loop instead:
//!
//! ```
//! let v = vec![1, 2, 3, 4, 5];
//!
//! for x in &v {
//!     println!("{}", x);
//! }
//! ```
//!
//! [`map`]: trait.Iterator.html#method.map
//!
//! The two most common ways to evaluate an iterator are to use a `for` loop
//! like this, or using the [`collect`] method to produce a new collection.
//!
//! [`collect`]: trait.Iterator.html#method.collect
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
//! result can be determined mathematically in finite time, may not terminate.
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
//! [`take`]: trait.Iterator.html#method.take
//! [`min`]: trait.Iterator.html#method.min

#![stable(feature = "rust1", since = "1.0.0")]

use crate::ops::Try;

#[stable(feature = "rust1", since = "1.0.0")]
pub use self::traits::Iterator;

#[unstable(feature = "step_trait",
           reason = "likely to be replaced by finer-grained traits",
           issue = "42168")]
pub use self::range::Step;

#[stable(feature = "rust1", since = "1.0.0")]
pub use self::sources::{Repeat, repeat};
#[stable(feature = "iterator_repeat_with", since = "1.28.0")]
pub use self::sources::{RepeatWith, repeat_with};
#[stable(feature = "iter_empty", since = "1.2.0")]
pub use self::sources::{Empty, empty};
#[stable(feature = "iter_once", since = "1.2.0")]
pub use self::sources::{Once, once};
#[unstable(feature = "iter_once_with", issue = "57581")]
pub use self::sources::{OnceWith, once_with};
#[stable(feature = "iter_from_fn", since = "1.34.0")]
pub use self::sources::{FromFn, from_fn};
#[stable(feature = "iter_successors", since = "1.34.0")]
pub use self::sources::{Successors, successors};

#[stable(feature = "rust1", since = "1.0.0")]
pub use self::traits::{FromIterator, IntoIterator, DoubleEndedIterator, Extend};
#[stable(feature = "rust1", since = "1.0.0")]
pub use self::traits::{ExactSizeIterator, Sum, Product};
#[stable(feature = "fused", since = "1.26.0")]
pub use self::traits::FusedIterator;
#[unstable(feature = "trusted_len", issue = "37572")]
pub use self::traits::TrustedLen;

#[stable(feature = "rust1", since = "1.0.0")]
pub use self::adapters::{Rev, Cycle, Chain, Zip, Map, Filter, FilterMap, Enumerate};
#[stable(feature = "rust1", since = "1.0.0")]
pub use self::adapters::{Peekable, SkipWhile, TakeWhile, Skip, Take, Scan, FlatMap};
#[stable(feature = "rust1", since = "1.0.0")]
pub use self::adapters::{Fuse, Inspect};
#[stable(feature = "iter_cloned", since = "1.1.0")]
pub use self::adapters::Cloned;
#[stable(feature = "iterator_step_by", since = "1.28.0")]
pub use self::adapters::StepBy;
#[stable(feature = "iterator_flatten", since = "1.29.0")]
pub use self::adapters::Flatten;
#[stable(feature = "iter_copied", since = "1.36.0")]
pub use self::adapters::Copied;

pub(crate) use self::adapters::TrustedRandomAccess;

mod range;
mod sources;
mod traits;
mod adapters;

/// Used to make try_fold closures more like normal loops
#[derive(PartialEq)]
enum LoopState<C, B> {
    Continue(C),
    Break(B),
}

impl<C, B> Try for LoopState<C, B> {
    type Ok = C;
    type Error = B;
    #[inline]
    fn into_result(self) -> Result<Self::Ok, Self::Error> {
        match self {
            LoopState::Continue(y) => Ok(y),
            LoopState::Break(x) => Err(x),
        }
    }
    #[inline]
    fn from_error(v: Self::Error) -> Self { LoopState::Break(v) }
    #[inline]
    fn from_ok(v: Self::Ok) -> Self { LoopState::Continue(v) }
}

impl<C, B> LoopState<C, B> {
    #[inline]
    fn break_value(self) -> Option<B> {
        match self {
            LoopState::Continue(..) => None,
            LoopState::Break(x) => Some(x),
        }
    }
}

impl<R: Try> LoopState<R::Ok, R> {
    #[inline]
    fn from_try(r: R) -> Self {
        match Try::into_result(r) {
            Ok(v) => LoopState::Continue(v),
            Err(v) => LoopState::Break(Try::from_error(v)),
        }
    }
    #[inline]
    fn into_try(self) -> R {
        match self {
            LoopState::Continue(v) => Try::from_ok(v),
            LoopState::Break(v) => v,
        }
    }
}
