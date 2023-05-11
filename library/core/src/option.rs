//! Optional values.
//!
//! Type [`Option`] represents an optional value: every [`Option`]
//! is either [`Some`] and contains a value, or [`None`], and
//! does not. [`Option`] types are very common in Rust code, as
//! they have a number of uses:
//!
//! * Initial values
//! * Return values for functions that are not defined
//!   over their entire input range (partial functions)
//! * Return value for otherwise reporting simple errors, where [`None`] is
//!   returned on error
//! * Optional struct fields
//! * Struct fields that can be loaned or "taken"
//! * Optional function arguments
//! * Nullable pointers
//! * Swapping things out of difficult situations
//!
//! [`Option`]s are commonly paired with pattern matching to query the presence
//! of a value and take action, always accounting for the [`None`] case.
//!
//! ```
//! fn divide(numerator: f64, denominator: f64) -> Option<f64> {
//!     if denominator == 0.0 {
//!         None
//!     } else {
//!         Some(numerator / denominator)
//!     }
//! }
//!
//! // The return value of the function is an option
//! let result = divide(2.0, 3.0);
//!
//! // Pattern match to retrieve the value
//! match result {
//!     // The division was valid
//!     Some(x) => println!("Result: {x}"),
//!     // The division was invalid
//!     None    => println!("Cannot divide by 0"),
//! }
//! ```
//!
//
// FIXME: Show how `Option` is used in practice, with lots of methods
//
//! # Options and pointers ("nullable" pointers)
//!
//! Rust's pointer types must always point to a valid location; there are
//! no "null" references. Instead, Rust has *optional* pointers, like
//! the optional owned box, <code>[Option]<[Box\<T>]></code>.
//!
//! [Box\<T>]: ../../std/boxed/struct.Box.html
//!
//! The following example uses [`Option`] to create an optional box of
//! [`i32`]. Notice that in order to use the inner [`i32`] value, the
//! `check_optional` function first needs to use pattern matching to
//! determine whether the box has a value (i.e., it is [`Some(...)`][`Some`]) or
//! not ([`None`]).
//!
//! ```
//! let optional = None;
//! check_optional(optional);
//!
//! let optional = Some(Box::new(9000));
//! check_optional(optional);
//!
//! fn check_optional(optional: Option<Box<i32>>) {
//!     match optional {
//!         Some(p) => println!("has value {p}"),
//!         None => println!("has no value"),
//!     }
//! }
//! ```
//!
//! # The question mark operator, `?`
//!
//! Similar to the [`Result`] type, when writing code that calls many functions that return the
//! [`Option`] type, handling `Some`/`None` can be tedious. The question mark
//! operator, [`?`], hides some of the boilerplate of propagating values
//! up the call stack.
//!
//! It replaces this:
//!
//! ```
//! # #![allow(dead_code)]
//! fn add_last_numbers(stack: &mut Vec<i32>) -> Option<i32> {
//!     let a = stack.pop();
//!     let b = stack.pop();
//!
//!     match (a, b) {
//!         (Some(x), Some(y)) => Some(x + y),
//!         _ => None,
//!     }
//! }
//!
//! ```
//!
//! With this:
//!
//! ```
//! # #![allow(dead_code)]
//! fn add_last_numbers(stack: &mut Vec<i32>) -> Option<i32> {
//!     Some(stack.pop()? + stack.pop()?)
//! }
//! ```
//!
//! *It's much nicer!*
//!
//! Ending the expression with [`?`] will result in the [`Some`]'s unwrapped value, unless the
//! result is [`None`], in which case [`None`] is returned early from the enclosing function.
//!
//! [`?`] can be used in functions that return [`Option`] because of the
//! early return of [`None`] that it provides.
//!
//! [`?`]: crate::ops::Try
//! [`Some`]: Some
//! [`None`]: None
//!
//! # Representation
//!
//! Rust guarantees to optimize the following types `T` such that
//! [`Option<T>`] has the same size as `T`:
//!
//! * [`Box<U>`]
//! * `&U`
//! * `&mut U`
//! * `fn`, `extern "C" fn`[^extern_fn]
//! * [`num::NonZero*`]
//! * [`ptr::NonNull<U>`]
//! * `#[repr(transparent)]` struct around one of the types in this list.
//!
//! [^extern_fn]: this remains true for any other ABI: `extern "abi" fn` (_e.g._, `extern "system" fn`)
//!
//! [`Box<U>`]: ../../std/boxed/struct.Box.html
//! [`num::NonZero*`]: crate::num
//! [`ptr::NonNull<U>`]: crate::ptr::NonNull
//!
//! This is called the "null pointer optimization" or NPO.
//!
//! It is further guaranteed that, for the cases above, one can
//! [`mem::transmute`] from all valid values of `T` to `Option<T>` and
//! from `Some::<T>(_)` to `T` (but transmuting `None::<T>` to `T`
//! is undefined behaviour).
//!
//! # Method overview
//!
//! In addition to working with pattern matching, [`Option`] provides a wide
//! variety of different methods.
//!
//! ## Querying the variant
//!
//! The [`is_some`] and [`is_none`] methods return [`true`] if the [`Option`]
//! is [`Some`] or [`None`], respectively.
//!
//! [`is_none`]: Option::is_none
//! [`is_some`]: Option::is_some
//!
//! ## Adapters for working with references
//!
//! * [`as_ref`] converts from <code>[&][][Option]\<T></code> to <code>[Option]<[&]T></code>
//! * [`as_mut`] converts from <code>[&mut] [Option]\<T></code> to <code>[Option]<[&mut] T></code>
//! * [`as_deref`] converts from <code>[&][][Option]\<T></code> to
//!   <code>[Option]<[&]T::[Target]></code>
//! * [`as_deref_mut`] converts from <code>[&mut] [Option]\<T></code> to
//!   <code>[Option]<[&mut] T::[Target]></code>
//! * [`as_pin_ref`] converts from <code>[Pin]<[&][][Option]\<T>></code> to
//!   <code>[Option]<[Pin]<[&]T>></code>
//! * [`as_pin_mut`] converts from <code>[Pin]<[&mut] [Option]\<T>></code> to
//!   <code>[Option]<[Pin]<[&mut] T>></code>
//!
//! [&]: reference "shared reference"
//! [&mut]: reference "mutable reference"
//! [Target]: Deref::Target "ops::Deref::Target"
//! [`as_deref`]: Option::as_deref
//! [`as_deref_mut`]: Option::as_deref_mut
//! [`as_mut`]: Option::as_mut
//! [`as_pin_mut`]: Option::as_pin_mut
//! [`as_pin_ref`]: Option::as_pin_ref
//! [`as_ref`]: Option::as_ref
//!
//! ## Extracting the contained value
//!
//! These methods extract the contained value in an [`Option<T>`] when it
//! is the [`Some`] variant. If the [`Option`] is [`None`]:
//!
//! * [`expect`] panics with a provided custom message
//! * [`unwrap`] panics with a generic message
//! * [`unwrap_or`] returns the provided default value
//! * [`unwrap_or_default`] returns the default value of the type `T`
//!   (which must implement the [`Default`] trait)
//! * [`unwrap_or_else`] returns the result of evaluating the provided
//!   function
//!
//! [`expect`]: Option::expect
//! [`unwrap`]: Option::unwrap
//! [`unwrap_or`]: Option::unwrap_or
//! [`unwrap_or_default`]: Option::unwrap_or_default
//! [`unwrap_or_else`]: Option::unwrap_or_else
//!
//! ## Transforming contained values
//!
//! These methods transform [`Option`] to [`Result`]:
//!
//! * [`ok_or`] transforms [`Some(v)`] to [`Ok(v)`], and [`None`] to
//!   [`Err(err)`] using the provided default `err` value
//! * [`ok_or_else`] transforms [`Some(v)`] to [`Ok(v)`], and [`None`] to
//!   a value of [`Err`] using the provided function
//! * [`transpose`] transposes an [`Option`] of a [`Result`] into a
//!   [`Result`] of an [`Option`]
//!
//! [`Err(err)`]: Err
//! [`Ok(v)`]: Ok
//! [`Some(v)`]: Some
//! [`ok_or`]: Option::ok_or
//! [`ok_or_else`]: Option::ok_or_else
//! [`transpose`]: Option::transpose
//!
//! These methods transform the [`Some`] variant:
//!
//! * [`filter`] calls the provided predicate function on the contained
//!   value `t` if the [`Option`] is [`Some(t)`], and returns [`Some(t)`]
//!   if the function returns `true`; otherwise, returns [`None`]
//! * [`flatten`] removes one level of nesting from an
//!   [`Option<Option<T>>`]
//! * [`map`] transforms [`Option<T>`] to [`Option<U>`] by applying the
//!   provided function to the contained value of [`Some`] and leaving
//!   [`None`] values unchanged
//!
//! [`Some(t)`]: Some
//! [`filter`]: Option::filter
//! [`flatten`]: Option::flatten
//! [`map`]: Option::map
//!
//! These methods transform [`Option<T>`] to a value of a possibly
//! different type `U`:
//!
//! * [`map_or`] applies the provided function to the contained value of
//!   [`Some`], or returns the provided default value if the [`Option`] is
//!   [`None`]
//! * [`map_or_else`] applies the provided function to the contained value
//!   of [`Some`], or returns the result of evaluating the provided
//!   fallback function if the [`Option`] is [`None`]
//!
//! [`map_or`]: Option::map_or
//! [`map_or_else`]: Option::map_or_else
//!
//! These methods combine the [`Some`] variants of two [`Option`] values:
//!
//! * [`zip`] returns [`Some((s, o))`] if `self` is [`Some(s)`] and the
//!   provided [`Option`] value is [`Some(o)`]; otherwise, returns [`None`]
//! * [`zip_with`] calls the provided function `f` and returns
//!   [`Some(f(s, o))`] if `self` is [`Some(s)`] and the provided
//!   [`Option`] value is [`Some(o)`]; otherwise, returns [`None`]
//!
//! [`Some(f(s, o))`]: Some
//! [`Some(o)`]: Some
//! [`Some(s)`]: Some
//! [`Some((s, o))`]: Some
//! [`zip`]: Option::zip
//! [`zip_with`]: Option::zip_with
//!
//! ## Boolean operators
//!
//! These methods treat the [`Option`] as a boolean value, where [`Some`]
//! acts like [`true`] and [`None`] acts like [`false`]. There are two
//! categories of these methods: ones that take an [`Option`] as input, and
//! ones that take a function as input (to be lazily evaluated).
//!
//! The [`and`], [`or`], and [`xor`] methods take another [`Option`] as
//! input, and produce an [`Option`] as output. Only the [`and`] method can
//! produce an [`Option<U>`] value having a different inner type `U` than
//! [`Option<T>`].
//!
//! | method  | self      | input     | output    |
//! |---------|-----------|-----------|-----------|
//! | [`and`] | `None`    | (ignored) | `None`    |
//! | [`and`] | `Some(x)` | `None`    | `None`    |
//! | [`and`] | `Some(x)` | `Some(y)` | `Some(y)` |
//! | [`or`]  | `None`    | `None`    | `None`    |
//! | [`or`]  | `None`    | `Some(y)` | `Some(y)` |
//! | [`or`]  | `Some(x)` | (ignored) | `Some(x)` |
//! | [`xor`] | `None`    | `None`    | `None`    |
//! | [`xor`] | `None`    | `Some(y)` | `Some(y)` |
//! | [`xor`] | `Some(x)` | `None`    | `Some(x)` |
//! | [`xor`] | `Some(x)` | `Some(y)` | `None`    |
//!
//! [`and`]: Option::and
//! [`or`]: Option::or
//! [`xor`]: Option::xor
//!
//! The [`and_then`] and [`or_else`] methods take a function as input, and
//! only evaluate the function when they need to produce a new value. Only
//! the [`and_then`] method can produce an [`Option<U>`] value having a
//! different inner type `U` than [`Option<T>`].
//!
//! | method       | self      | function input | function result | output    |
//! |--------------|-----------|----------------|-----------------|-----------|
//! | [`and_then`] | `None`    | (not provided) | (not evaluated) | `None`    |
//! | [`and_then`] | `Some(x)` | `x`            | `None`          | `None`    |
//! | [`and_then`] | `Some(x)` | `x`            | `Some(y)`       | `Some(y)` |
//! | [`or_else`]  | `None`    | (not provided) | `None`          | `None`    |
//! | [`or_else`]  | `None`    | (not provided) | `Some(y)`       | `Some(y)` |
//! | [`or_else`]  | `Some(x)` | (not provided) | (not evaluated) | `Some(x)` |
//!
//! [`and_then`]: Option::and_then
//! [`or_else`]: Option::or_else
//!
//! This is an example of using methods like [`and_then`] and [`or`] in a
//! pipeline of method calls. Early stages of the pipeline pass failure
//! values ([`None`]) through unchanged, and continue processing on
//! success values ([`Some`]). Toward the end, [`or`] substitutes an error
//! message if it receives [`None`].
//!
//! ```
//! # use std::collections::BTreeMap;
//! let mut bt = BTreeMap::new();
//! bt.insert(20u8, "foo");
//! bt.insert(42u8, "bar");
//! let res = [0u8, 1, 11, 200, 22]
//!     .into_iter()
//!     .map(|x| {
//!         // `checked_sub()` returns `None` on error
//!         x.checked_sub(1)
//!             // same with `checked_mul()`
//!             .and_then(|x| x.checked_mul(2))
//!             // `BTreeMap::get` returns `None` on error
//!             .and_then(|x| bt.get(&x))
//!             // Substitute an error message if we have `None` so far
//!             .or(Some(&"error!"))
//!             .copied()
//!             // Won't panic because we unconditionally used `Some` above
//!             .unwrap()
//!     })
//!     .collect::<Vec<_>>();
//! assert_eq!(res, ["error!", "error!", "foo", "error!", "bar"]);
//! ```
//!
//! ## Comparison operators
//!
//! If `T` implements [`PartialOrd`] then [`Option<T>`] will derive its
//! [`PartialOrd`] implementation.  With this order, [`None`] compares as
//! less than any [`Some`], and two [`Some`] compare the same way as their
//! contained values would in `T`.  If `T` also implements
//! [`Ord`], then so does [`Option<T>`].
//!
//! ```
//! assert!(None < Some(0));
//! assert!(Some(0) < Some(1));
//! ```
//!
//! ## Iterating over `Option`
//!
//! An [`Option`] can be iterated over. This can be helpful if you need an
//! iterator that is conditionally empty. The iterator will either produce
//! a single value (when the [`Option`] is [`Some`]), or produce no values
//! (when the [`Option`] is [`None`]). For example, [`into_iter`] acts like
//! [`once(v)`] if the [`Option`] is [`Some(v)`], and like [`empty()`] if
//! the [`Option`] is [`None`].
//!
//! [`Some(v)`]: Some
//! [`empty()`]: crate::iter::empty
//! [`once(v)`]: crate::iter::once
//!
//! Iterators over [`Option<T>`] come in three types:
//!
//! * [`into_iter`] consumes the [`Option`] and produces the contained
//!   value
//! * [`iter`] produces an immutable reference of type `&T` to the
//!   contained value
//! * [`iter_mut`] produces a mutable reference of type `&mut T` to the
//!   contained value
//!
//! [`into_iter`]: Option::into_iter
//! [`iter`]: Option::iter
//! [`iter_mut`]: Option::iter_mut
//!
//! An iterator over [`Option`] can be useful when chaining iterators, for
//! example, to conditionally insert items. (It's not always necessary to
//! explicitly call an iterator constructor: many [`Iterator`] methods that
//! accept other iterators will also accept iterable types that implement
//! [`IntoIterator`], which includes [`Option`].)
//!
//! ```
//! let yep = Some(42);
//! let nope = None;
//! // chain() already calls into_iter(), so we don't have to do so
//! let nums: Vec<i32> = (0..4).chain(yep).chain(4..8).collect();
//! assert_eq!(nums, [0, 1, 2, 3, 42, 4, 5, 6, 7]);
//! let nums: Vec<i32> = (0..4).chain(nope).chain(4..8).collect();
//! assert_eq!(nums, [0, 1, 2, 3, 4, 5, 6, 7]);
//! ```
//!
//! One reason to chain iterators in this way is that a function returning
//! `impl Iterator` must have all possible return values be of the same
//! concrete type. Chaining an iterated [`Option`] can help with that.
//!
//! ```
//! fn make_iter(do_insert: bool) -> impl Iterator<Item = i32> {
//!     // Explicit returns to illustrate return types matching
//!     match do_insert {
//!         true => return (0..4).chain(Some(42)).chain(4..8),
//!         false => return (0..4).chain(None).chain(4..8),
//!     }
//! }
//! println!("{:?}", make_iter(true).collect::<Vec<_>>());
//! println!("{:?}", make_iter(false).collect::<Vec<_>>());
//! ```
//!
//! If we try to do the same thing, but using [`once()`] and [`empty()`],
//! we can't return `impl Iterator` anymore because the concrete types of
//! the return values differ.
//!
//! [`empty()`]: crate::iter::empty
//! [`once()`]: crate::iter::once
//!
//! ```compile_fail,E0308
//! # use std::iter::{empty, once};
//! // This won't compile because all possible returns from the function
//! // must have the same concrete type.
//! fn make_iter(do_insert: bool) -> impl Iterator<Item = i32> {
//!     // Explicit returns to illustrate return types not matching
//!     match do_insert {
//!         true => return (0..4).chain(once(42)).chain(4..8),
//!         false => return (0..4).chain(empty()).chain(4..8),
//!     }
//! }
//! ```
//!
//! ## Collecting into `Option`
//!
//! [`Option`] implements the [`FromIterator`][impl-FromIterator] trait,
//! which allows an iterator over [`Option`] values to be collected into an
//! [`Option`] of a collection of each contained value of the original
//! [`Option`] values, or [`None`] if any of the elements was [`None`].
//!
//! [impl-FromIterator]: Option#impl-FromIterator%3COption%3CA%3E%3E-for-Option%3CV%3E
//!
//! ```
//! let v = [Some(2), Some(4), None, Some(8)];
//! let res: Option<Vec<_>> = v.into_iter().collect();
//! assert_eq!(res, None);
//! let v = [Some(2), Some(4), Some(8)];
//! let res: Option<Vec<_>> = v.into_iter().collect();
//! assert_eq!(res, Some(vec![2, 4, 8]));
//! ```
//!
//! [`Option`] also implements the [`Product`][impl-Product] and
//! [`Sum`][impl-Sum] traits, allowing an iterator over [`Option`] values
//! to provide the [`product`][Iterator::product] and
//! [`sum`][Iterator::sum] methods.
//!
//! [impl-Product]: Option#impl-Product%3COption%3CU%3E%3E-for-Option%3CT%3E
//! [impl-Sum]: Option#impl-Sum%3COption%3CU%3E%3E-for-Option%3CT%3E
//!
//! ```
//! let v = [None, Some(1), Some(2), Some(3)];
//! let res: Option<i32> = v.into_iter().sum();
//! assert_eq!(res, None);
//! let v = [Some(1), Some(2), Some(21)];
//! let res: Option<i32> = v.into_iter().product();
//! assert_eq!(res, Some(42));
//! ```
//!
//! ## Modifying an [`Option`] in-place
//!
//! These methods return a mutable reference to the contained value of an
//! [`Option<T>`]:
//!
//! * [`insert`] inserts a value, dropping any old contents
//! * [`get_or_insert`] gets the current value, inserting a provided
//!   default value if it is [`None`]
//! * [`get_or_insert_default`] gets the current value, inserting the
//!   default value of type `T` (which must implement [`Default`]) if it is
//!   [`None`]
//! * [`get_or_insert_with`] gets the current value, inserting a default
//!   computed by the provided function if it is [`None`]
//!
//! [`get_or_insert`]: Option::get_or_insert
//! [`get_or_insert_default`]: Option::get_or_insert_default
//! [`get_or_insert_with`]: Option::get_or_insert_with
//! [`insert`]: Option::insert
//!
//! These methods transfer ownership of the contained value of an
//! [`Option`]:
//!
//! * [`take`] takes ownership of the contained value of an [`Option`], if
//!   any, replacing the [`Option`] with [`None`]
//! * [`replace`] takes ownership of the contained value of an [`Option`],
//!   if any, replacing the [`Option`] with a [`Some`] containing the
//!   provided value
//!
//! [`replace`]: Option::replace
//! [`take`]: Option::take
//!
//! # Examples
//!
//! Basic pattern matching on [`Option`]:
//!
//! ```
//! let msg = Some("howdy");
//!
//! // Take a reference to the contained string
//! if let Some(m) = &msg {
//!     println!("{}", *m);
//! }
//!
//! // Remove the contained string, destroying the Option
//! let unwrapped_msg = msg.unwrap_or("default message");
//! ```
//!
//! Initialize a result to [`None`] before a loop:
//!
//! ```
//! enum Kingdom { Plant(u32, &'static str), Animal(u32, &'static str) }
//!
//! // A list of data to search through.
//! let all_the_big_things = [
//!     Kingdom::Plant(250, "redwood"),
//!     Kingdom::Plant(230, "noble fir"),
//!     Kingdom::Plant(229, "sugar pine"),
//!     Kingdom::Animal(25, "blue whale"),
//!     Kingdom::Animal(19, "fin whale"),
//!     Kingdom::Animal(15, "north pacific right whale"),
//! ];
//!
//! // We're going to search for the name of the biggest animal,
//! // but to start with we've just got `None`.
//! let mut name_of_biggest_animal = None;
//! let mut size_of_biggest_animal = 0;
//! for big_thing in &all_the_big_things {
//!     match *big_thing {
//!         Kingdom::Animal(size, name) if size > size_of_biggest_animal => {
//!             // Now we've found the name of some big animal
//!             size_of_biggest_animal = size;
//!             name_of_biggest_animal = Some(name);
//!         }
//!         Kingdom::Animal(..) | Kingdom::Plant(..) => ()
//!     }
//! }
//!
//! match name_of_biggest_animal {
//!     Some(name) => println!("the biggest animal is {name}"),
//!     None => println!("there are no animals :("),
//! }
//! ```

#![stable(feature = "rust1", since = "1.0.0")]

use crate::iter::{self, FromIterator, FusedIterator, TrustedLen};
use crate::panicking::{panic, panic_str};
use crate::pin::Pin;
use crate::{
    cmp, convert, hint, mem,
    ops::{self, ControlFlow, Deref, DerefMut},
    slice,
};

/// The `Option` type. See [the module level documentation](self) for more.
#[derive(Copy, PartialOrd, Eq, Ord, Debug, Hash)]
#[rustc_diagnostic_item = "Option"]
#[lang = "Option"]
#[stable(feature = "rust1", since = "1.0.0")]
pub enum Option<T> {
    /// No value.
    #[lang = "None"]
    #[stable(feature = "rust1", since = "1.0.0")]
    None,
    /// Some value of type `T`.
    #[lang = "Some"]
    #[stable(feature = "rust1", since = "1.0.0")]
    Some(#[stable(feature = "rust1", since = "1.0.0")] T),
}

/////////////////////////////////////////////////////////////////////////////
// Type implementation
/////////////////////////////////////////////////////////////////////////////

impl<T> Option<T> {
    /////////////////////////////////////////////////////////////////////////
    // Querying the contained values
    /////////////////////////////////////////////////////////////////////////

    /// Returns `true` if the option is a [`Some`] value.
    ///
    /// # Examples
    ///
    /// ```
    /// let x: Option<u32> = Some(2);
    /// assert_eq!(x.is_some(), true);
    ///
    /// let x: Option<u32> = None;
    /// assert_eq!(x.is_some(), false);
    /// ```
    #[must_use = "if you intended to assert that this has a value, consider `.unwrap()` instead"]
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_stable(feature = "const_option_basics", since = "1.48.0")]
    pub const fn is_some(&self) -> bool {
        matches!(*self, Some(_))
    }

    /// Returns `true` if the option is a [`Some`] and the value inside of it matches a predicate.
    ///
    /// # Examples
    ///
    /// ```
    /// let x: Option<u32> = Some(2);
    /// assert_eq!(x.is_some_and(|x| x > 1), true);
    ///
    /// let x: Option<u32> = Some(0);
    /// assert_eq!(x.is_some_and(|x| x > 1), false);
    ///
    /// let x: Option<u32> = None;
    /// assert_eq!(x.is_some_and(|x| x > 1), false);
    /// ```
    #[must_use]
    #[inline]
    #[stable(feature = "is_some_and", since = "1.70.0")]
    pub fn is_some_and(self, f: impl FnOnce(T) -> bool) -> bool {
        match self {
            None => false,
            Some(x) => f(x),
        }
    }

    /// Returns `true` if the option is a [`None`] value.
    ///
    /// # Examples
    ///
    /// ```
    /// let x: Option<u32> = Some(2);
    /// assert_eq!(x.is_none(), false);
    ///
    /// let x: Option<u32> = None;
    /// assert_eq!(x.is_none(), true);
    /// ```
    #[must_use = "if you intended to assert that this doesn't have a value, consider \
                  `.and_then(|_| panic!(\"`Option` had a value when expected `None`\"))` instead"]
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_stable(feature = "const_option_basics", since = "1.48.0")]
    pub const fn is_none(&self) -> bool {
        !self.is_some()
    }

    /////////////////////////////////////////////////////////////////////////
    // Adapter for working with references
    /////////////////////////////////////////////////////////////////////////

    /// Converts from `&Option<T>` to `Option<&T>`.
    ///
    /// # Examples
    ///
    /// Calculates the length of an <code>Option<[String]></code> as an <code>Option<[usize]></code>
    /// without moving the [`String`]. The [`map`] method takes the `self` argument by value,
    /// consuming the original, so this technique uses `as_ref` to first take an `Option` to a
    /// reference to the value inside the original.
    ///
    /// [`map`]: Option::map
    /// [String]: ../../std/string/struct.String.html "String"
    /// [`String`]: ../../std/string/struct.String.html "String"
    ///
    /// ```
    /// let text: Option<String> = Some("Hello, world!".to_string());
    /// // First, cast `Option<String>` to `Option<&String>` with `as_ref`,
    /// // then consume *that* with `map`, leaving `text` on the stack.
    /// let text_length: Option<usize> = text.as_ref().map(|s| s.len());
    /// println!("still can print text: {text:?}");
    /// ```
    #[inline]
    #[rustc_const_stable(feature = "const_option_basics", since = "1.48.0")]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const fn as_ref(&self) -> Option<&T> {
        match *self {
            Some(ref x) => Some(x),
            None => None,
        }
    }

    /// Converts from `&mut Option<T>` to `Option<&mut T>`.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut x = Some(2);
    /// match x.as_mut() {
    ///     Some(v) => *v = 42,
    ///     None => {},
    /// }
    /// assert_eq!(x, Some(42));
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_unstable(feature = "const_option", issue = "67441")]
    pub const fn as_mut(&mut self) -> Option<&mut T> {
        match *self {
            Some(ref mut x) => Some(x),
            None => None,
        }
    }

    /// Converts from <code>[Pin]<[&]Option\<T>></code> to <code>Option<[Pin]<[&]T>></code>.
    ///
    /// [&]: reference "shared reference"
    #[inline]
    #[must_use]
    #[stable(feature = "pin", since = "1.33.0")]
    #[rustc_const_unstable(feature = "const_option_ext", issue = "91930")]
    pub const fn as_pin_ref(self: Pin<&Self>) -> Option<Pin<&T>> {
        match Pin::get_ref(self).as_ref() {
            // SAFETY: `x` is guaranteed to be pinned because it comes from `self`
            // which is pinned.
            Some(x) => unsafe { Some(Pin::new_unchecked(x)) },
            None => None,
        }
    }

    /// Converts from <code>[Pin]<[&mut] Option\<T>></code> to <code>Option<[Pin]<[&mut] T>></code>.
    ///
    /// [&mut]: reference "mutable reference"
    #[inline]
    #[must_use]
    #[stable(feature = "pin", since = "1.33.0")]
    #[rustc_const_unstable(feature = "const_option_ext", issue = "91930")]
    pub const fn as_pin_mut(self: Pin<&mut Self>) -> Option<Pin<&mut T>> {
        // SAFETY: `get_unchecked_mut` is never used to move the `Option` inside `self`.
        // `x` is guaranteed to be pinned because it comes from `self` which is pinned.
        unsafe {
            match Pin::get_unchecked_mut(self).as_mut() {
                Some(x) => Some(Pin::new_unchecked(x)),
                None => None,
            }
        }
    }

    /// Returns a slice of the contained value, if any. If this is `None`, an
    /// empty slice is returned. This can be useful to have a single type of
    /// iterator over an `Option` or slice.
    ///
    /// Note: Should you have an `Option<&T>` and wish to get a slice of `T`,
    /// you can unpack it via `opt.map_or(&[], std::slice::from_ref)`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// #![feature(option_as_slice)]
    ///
    /// assert_eq!(
    ///     [Some(1234).as_slice(), None.as_slice()],
    ///     [&[1234][..], &[][..]],
    /// );
    /// ```
    ///
    /// The inverse of this function is (discounting
    /// borrowing) [`[_]::first`](slice::first):
    ///
    /// ```rust
    /// #![feature(option_as_slice)]
    ///
    /// for i in [Some(1234_u16), None] {
    ///     assert_eq!(i.as_ref(), i.as_slice().first());
    /// }
    /// ```
    #[inline]
    #[must_use]
    #[unstable(feature = "option_as_slice", issue = "108545")]
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: When the `Option` is `Some`, we're using the actual pointer
        // to the payload, with a length of 1, so this is equivalent to
        // `slice::from_ref`, and thus is safe.
        // When the `Option` is `None`, the length used is 0, so to be safe it
        // just needs to be aligned, which it is because `&self` is aligned and
        // the offset used is a multiple of alignment.
        //
        // In the new version, the intrinsic always returns a pointer to an
        // in-bounds and correctly aligned position for a `T` (even if in the
        // `None` case it's just padding).
        unsafe {
            slice::from_raw_parts(
                crate::intrinsics::option_payload_ptr(crate::ptr::from_ref(self)),
                usize::from(self.is_some()),
            )
        }
    }

    /// Returns a mutable slice of the contained value, if any. If this is
    /// `None`, an empty slice is returned. This can be useful to have a
    /// single type of iterator over an `Option` or slice.
    ///
    /// Note: Should you have an `Option<&mut T>` instead of a
    /// `&mut Option<T>`, which this method takes, you can obtain a mutable
    /// slice via `opt.map_or(&mut [], std::slice::from_mut)`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// #![feature(option_as_slice)]
    ///
    /// assert_eq!(
    ///     [Some(1234).as_mut_slice(), None.as_mut_slice()],
    ///     [&mut [1234][..], &mut [][..]],
    /// );
    /// ```
    ///
    /// The result is a mutable slice of zero or one items that points into
    /// our original `Option`:
    ///
    /// ```rust
    /// #![feature(option_as_slice)]
    ///
    /// let mut x = Some(1234);
    /// x.as_mut_slice()[0] += 1;
    /// assert_eq!(x, Some(1235));
    /// ```
    ///
    /// The inverse of this method (discounting borrowing)
    /// is [`[_]::first_mut`](slice::first_mut):
    ///
    /// ```rust
    /// #![feature(option_as_slice)]
    ///
    /// assert_eq!(Some(123).as_mut_slice().first_mut(), Some(&mut 123))
    /// ```
    #[inline]
    #[must_use]
    #[unstable(feature = "option_as_slice", issue = "108545")]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: When the `Option` is `Some`, we're using the actual pointer
        // to the payload, with a length of 1, so this is equivalent to
        // `slice::from_mut`, and thus is safe.
        // When the `Option` is `None`, the length used is 0, so to be safe it
        // just needs to be aligned, which it is because `&self` is aligned and
        // the offset used is a multiple of alignment.
        //
        // In the new version, the intrinsic creates a `*const T` from a
        // mutable reference  so it is safe to cast back to a mutable pointer
        // here. As with `as_slice`, the intrinsic always returns a pointer to
        // an in-bounds and correctly aligned position for a `T` (even if in
        // the `None` case it's just padding).
        unsafe {
            slice::from_raw_parts_mut(
                crate::intrinsics::option_payload_ptr(crate::ptr::from_mut(self).cast_const())
                    .cast_mut(),
                usize::from(self.is_some()),
            )
        }
    }

    /////////////////////////////////////////////////////////////////////////
    // Getting to contained values
    /////////////////////////////////////////////////////////////////////////

    /// Returns the contained [`Some`] value, consuming the `self` value.
    ///
    /// # Panics
    ///
    /// Panics if the value is a [`None`] with a custom panic message provided by
    /// `msg`.
    ///
    /// # Examples
    ///
    /// ```
    /// let x = Some("value");
    /// assert_eq!(x.expect("fruits are healthy"), "value");
    /// ```
    ///
    /// ```should_panic
    /// let x: Option<&str> = None;
    /// x.expect("fruits are healthy"); // panics with `fruits are healthy`
    /// ```
    ///
    /// # Recommended Message Style
    ///
    /// We recommend that `expect` messages are used to describe the reason you
    /// _expect_ the `Option` should be `Some`.
    ///
    /// ```should_panic
    /// # let slice: &[u8] = &[];
    /// let item = slice.get(0)
    ///     .expect("slice should not be empty");
    /// ```
    ///
    /// **Hint**: If you're having trouble remembering how to phrase expect
    /// error messages remember to focus on the word "should" as in "env
    /// variable should be set by blah" or "the given binary should be available
    /// and executable by the current user".
    ///
    /// For more detail on expect message styles and the reasoning behind our
    /// recommendation please refer to the section on ["Common Message
    /// Styles"](../../std/error/index.html#common-message-styles) in the [`std::error`](../../std/error/index.html) module docs.
    #[inline]
    #[track_caller]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_unstable(feature = "const_option", issue = "67441")]
    pub const fn expect(self, msg: &str) -> T {
        match self {
            Some(val) => val,
            None => expect_failed(msg),
        }
    }

    /// Returns the contained [`Some`] value, consuming the `self` value.
    ///
    /// Because this function may panic, its use is generally discouraged.
    /// Instead, prefer to use pattern matching and handle the [`None`]
    /// case explicitly, or call [`unwrap_or`], [`unwrap_or_else`], or
    /// [`unwrap_or_default`].
    ///
    /// [`unwrap_or`]: Option::unwrap_or
    /// [`unwrap_or_else`]: Option::unwrap_or_else
    /// [`unwrap_or_default`]: Option::unwrap_or_default
    ///
    /// # Panics
    ///
    /// Panics if the self value equals [`None`].
    ///
    /// # Examples
    ///
    /// ```
    /// let x = Some("air");
    /// assert_eq!(x.unwrap(), "air");
    /// ```
    ///
    /// ```should_panic
    /// let x: Option<&str> = None;
    /// assert_eq!(x.unwrap(), "air"); // fails
    /// ```
    #[inline]
    #[track_caller]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_unstable(feature = "const_option", issue = "67441")]
    pub const fn unwrap(self) -> T {
        match self {
            Some(val) => val,
            None => panic("called `Option::unwrap()` on a `None` value"),
        }
    }

    /// Returns the contained [`Some`] value or a provided default.
    ///
    /// Arguments passed to `unwrap_or` are eagerly evaluated; if you are passing
    /// the result of a function call, it is recommended to use [`unwrap_or_else`],
    /// which is lazily evaluated.
    ///
    /// [`unwrap_or_else`]: Option::unwrap_or_else
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!(Some("car").unwrap_or("bike"), "car");
    /// assert_eq!(None.unwrap_or("bike"), "bike");
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn unwrap_or(self, default: T) -> T {
        match self {
            Some(x) => x,
            None => default,
        }
    }

    /// Returns the contained [`Some`] value or computes it from a closure.
    ///
    /// # Examples
    ///
    /// ```
    /// let k = 10;
    /// assert_eq!(Some(4).unwrap_or_else(|| 2 * k), 4);
    /// assert_eq!(None.unwrap_or_else(|| 2 * k), 20);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn unwrap_or_else<F>(self, f: F) -> T
    where
        F: FnOnce() -> T,
    {
        match self {
            Some(x) => x,
            None => f(),
        }
    }

    /// Returns the contained [`Some`] value or a default.
    ///
    /// Consumes the `self` argument then, if [`Some`], returns the contained
    /// value, otherwise if [`None`], returns the [default value] for that
    /// type.
    ///
    /// # Examples
    ///
    /// ```
    /// let x: Option<u32> = None;
    /// let y: Option<u32> = Some(12);
    ///
    /// assert_eq!(x.unwrap_or_default(), 0);
    /// assert_eq!(y.unwrap_or_default(), 12);
    /// ```
    ///
    /// [default value]: Default::default
    /// [`parse`]: str::parse
    /// [`FromStr`]: crate::str::FromStr
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn unwrap_or_default(self) -> T
    where
        T: Default,
    {
        match self {
            Some(x) => x,
            None => T::default(),
        }
    }

    /// Returns the contained [`Some`] value, consuming the `self` value,
    /// without checking that the value is not [`None`].
    ///
    /// # Safety
    ///
    /// Calling this method on [`None`] is *[undefined behavior]*.
    ///
    /// [undefined behavior]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
    ///
    /// # Examples
    ///
    /// ```
    /// let x = Some("air");
    /// assert_eq!(unsafe { x.unwrap_unchecked() }, "air");
    /// ```
    ///
    /// ```no_run
    /// let x: Option<&str> = None;
    /// assert_eq!(unsafe { x.unwrap_unchecked() }, "air"); // Undefined behavior!
    /// ```
    #[inline]
    #[track_caller]
    #[stable(feature = "option_result_unwrap_unchecked", since = "1.58.0")]
    #[rustc_const_unstable(feature = "const_option_ext", issue = "91930")]
    pub const unsafe fn unwrap_unchecked(self) -> T {
        debug_assert!(self.is_some());
        match self {
            Some(val) => val,
            // SAFETY: the safety contract must be upheld by the caller.
            None => unsafe { hint::unreachable_unchecked() },
        }
    }

    /////////////////////////////////////////////////////////////////////////
    // Transforming contained values
    /////////////////////////////////////////////////////////////////////////

    /// Maps an `Option<T>` to `Option<U>` by applying a function to a contained value (if `Some`) or returns `None` (if `None`).
    ///
    /// # Examples
    ///
    /// Calculates the length of an <code>Option<[String]></code> as an
    /// <code>Option<[usize]></code>, consuming the original:
    ///
    /// [String]: ../../std/string/struct.String.html "String"
    /// ```
    /// let maybe_some_string = Some(String::from("Hello, World!"));
    /// // `Option::map` takes self *by value*, consuming `maybe_some_string`
    /// let maybe_some_len = maybe_some_string.map(|s| s.len());
    /// assert_eq!(maybe_some_len, Some(13));
    ///
    /// let x: Option<&str> = None;
    /// assert_eq!(x.map(|s| s.len()), None);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn map<U, F>(self, f: F) -> Option<U>
    where
        F: FnOnce(T) -> U,
    {
        match self {
            Some(x) => Some(f(x)),
            None => None,
        }
    }

    /// Calls the provided closure with a reference to the contained value (if [`Some`]).
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(result_option_inspect)]
    ///
    /// let v = vec![1, 2, 3, 4, 5];
    ///
    /// // prints "got: 4"
    /// let x: Option<&usize> = v.get(3).inspect(|x| println!("got: {x}"));
    ///
    /// // prints nothing
    /// let x: Option<&usize> = v.get(5).inspect(|x| println!("got: {x}"));
    /// ```
    #[inline]
    #[unstable(feature = "result_option_inspect", issue = "91345")]
    pub fn inspect<F>(self, f: F) -> Self
    where
        F: FnOnce(&T),
    {
        if let Some(ref x) = self {
            f(x);
        }

        self
    }

    /// Returns the provided default result (if none),
    /// or applies a function to the contained value (if any).
    ///
    /// Arguments passed to `map_or` are eagerly evaluated; if you are passing
    /// the result of a function call, it is recommended to use [`map_or_else`],
    /// which is lazily evaluated.
    ///
    /// [`map_or_else`]: Option::map_or_else
    ///
    /// # Examples
    ///
    /// ```
    /// let x = Some("foo");
    /// assert_eq!(x.map_or(42, |v| v.len()), 3);
    ///
    /// let x: Option<&str> = None;
    /// assert_eq!(x.map_or(42, |v| v.len()), 42);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn map_or<U, F>(self, default: U, f: F) -> U
    where
        F: FnOnce(T) -> U,
    {
        match self {
            Some(t) => f(t),
            None => default,
        }
    }

    /// Computes a default function result (if none), or
    /// applies a different function to the contained value (if any).
    ///
    /// # Examples
    ///
    /// ```
    /// let k = 21;
    ///
    /// let x = Some("foo");
    /// assert_eq!(x.map_or_else(|| 2 * k, |v| v.len()), 3);
    ///
    /// let x: Option<&str> = None;
    /// assert_eq!(x.map_or_else(|| 2 * k, |v| v.len()), 42);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn map_or_else<U, D, F>(self, default: D, f: F) -> U
    where
        D: FnOnce() -> U,
        F: FnOnce(T) -> U,
    {
        match self {
            Some(t) => f(t),
            None => default(),
        }
    }

    /// Transforms the `Option<T>` into a [`Result<T, E>`], mapping [`Some(v)`] to
    /// [`Ok(v)`] and [`None`] to [`Err(err)`].
    ///
    /// Arguments passed to `ok_or` are eagerly evaluated; if you are passing the
    /// result of a function call, it is recommended to use [`ok_or_else`], which is
    /// lazily evaluated.
    ///
    /// [`Ok(v)`]: Ok
    /// [`Err(err)`]: Err
    /// [`Some(v)`]: Some
    /// [`ok_or_else`]: Option::ok_or_else
    ///
    /// # Examples
    ///
    /// ```
    /// let x = Some("foo");
    /// assert_eq!(x.ok_or(0), Ok("foo"));
    ///
    /// let x: Option<&str> = None;
    /// assert_eq!(x.ok_or(0), Err(0));
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn ok_or<E>(self, err: E) -> Result<T, E> {
        match self {
            Some(v) => Ok(v),
            None => Err(err),
        }
    }

    /// Transforms the `Option<T>` into a [`Result<T, E>`], mapping [`Some(v)`] to
    /// [`Ok(v)`] and [`None`] to [`Err(err())`].
    ///
    /// [`Ok(v)`]: Ok
    /// [`Err(err())`]: Err
    /// [`Some(v)`]: Some
    ///
    /// # Examples
    ///
    /// ```
    /// let x = Some("foo");
    /// assert_eq!(x.ok_or_else(|| 0), Ok("foo"));
    ///
    /// let x: Option<&str> = None;
    /// assert_eq!(x.ok_or_else(|| 0), Err(0));
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn ok_or_else<E, F>(self, err: F) -> Result<T, E>
    where
        F: FnOnce() -> E,
    {
        match self {
            Some(v) => Ok(v),
            None => Err(err()),
        }
    }

    /// Converts from `Option<T>` (or `&Option<T>`) to `Option<&T::Target>`.
    ///
    /// Leaves the original Option in-place, creating a new one with a reference
    /// to the original one, additionally coercing the contents via [`Deref`].
    ///
    /// # Examples
    ///
    /// ```
    /// let x: Option<String> = Some("hey".to_owned());
    /// assert_eq!(x.as_deref(), Some("hey"));
    ///
    /// let x: Option<String> = None;
    /// assert_eq!(x.as_deref(), None);
    /// ```
    #[inline]
    #[stable(feature = "option_deref", since = "1.40.0")]
    pub fn as_deref(&self) -> Option<&T::Target>
    where
        T: Deref,
    {
        match self.as_ref() {
            Some(t) => Some(t.deref()),
            None => None,
        }
    }

    /// Converts from `Option<T>` (or `&mut Option<T>`) to `Option<&mut T::Target>`.
    ///
    /// Leaves the original `Option` in-place, creating a new one containing a mutable reference to
    /// the inner type's [`Deref::Target`] type.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut x: Option<String> = Some("hey".to_owned());
    /// assert_eq!(x.as_deref_mut().map(|x| {
    ///     x.make_ascii_uppercase();
    ///     x
    /// }), Some("HEY".to_owned().as_mut_str()));
    /// ```
    #[inline]
    #[stable(feature = "option_deref", since = "1.40.0")]
    pub fn as_deref_mut(&mut self) -> Option<&mut T::Target>
    where
        T: DerefMut,
    {
        match self.as_mut() {
            Some(t) => Some(t.deref_mut()),
            None => None,
        }
    }

    /////////////////////////////////////////////////////////////////////////
    // Iterator constructors
    /////////////////////////////////////////////////////////////////////////

    /// Returns an iterator over the possibly contained value.
    ///
    /// # Examples
    ///
    /// ```
    /// let x = Some(4);
    /// assert_eq!(x.iter().next(), Some(&4));
    ///
    /// let x: Option<u32> = None;
    /// assert_eq!(x.iter().next(), None);
    /// ```
    #[inline]
    #[rustc_const_unstable(feature = "const_option", issue = "67441")]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const fn iter(&self) -> Iter<'_, T> {
        Iter { inner: Item { opt: self.as_ref() } }
    }

    /// Returns a mutable iterator over the possibly contained value.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut x = Some(4);
    /// match x.iter_mut().next() {
    ///     Some(v) => *v = 42,
    ///     None => {},
    /// }
    /// assert_eq!(x, Some(42));
    ///
    /// let mut x: Option<u32> = None;
    /// assert_eq!(x.iter_mut().next(), None);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn iter_mut(&mut self) -> IterMut<'_, T> {
        IterMut { inner: Item { opt: self.as_mut() } }
    }

    /////////////////////////////////////////////////////////////////////////
    // Boolean operations on the values, eager and lazy
    /////////////////////////////////////////////////////////////////////////

    /// Returns [`None`] if the option is [`None`], otherwise returns `optb`.
    ///
    /// Arguments passed to `and` are eagerly evaluated; if you are passing the
    /// result of a function call, it is recommended to use [`and_then`], which is
    /// lazily evaluated.
    ///
    /// [`and_then`]: Option::and_then
    ///
    /// # Examples
    ///
    /// ```
    /// let x = Some(2);
    /// let y: Option<&str> = None;
    /// assert_eq!(x.and(y), None);
    ///
    /// let x: Option<u32> = None;
    /// let y = Some("foo");
    /// assert_eq!(x.and(y), None);
    ///
    /// let x = Some(2);
    /// let y = Some("foo");
    /// assert_eq!(x.and(y), Some("foo"));
    ///
    /// let x: Option<u32> = None;
    /// let y: Option<&str> = None;
    /// assert_eq!(x.and(y), None);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn and<U>(self, optb: Option<U>) -> Option<U> {
        match self {
            Some(_) => optb,
            None => None,
        }
    }

    /// Returns [`None`] if the option is [`None`], otherwise calls `f` with the
    /// wrapped value and returns the result.
    ///
    /// Some languages call this operation flatmap.
    ///
    /// # Examples
    ///
    /// ```
    /// fn sq_then_to_string(x: u32) -> Option<String> {
    ///     x.checked_mul(x).map(|sq| sq.to_string())
    /// }
    ///
    /// assert_eq!(Some(2).and_then(sq_then_to_string), Some(4.to_string()));
    /// assert_eq!(Some(1_000_000).and_then(sq_then_to_string), None); // overflowed!
    /// assert_eq!(None.and_then(sq_then_to_string), None);
    /// ```
    ///
    /// Often used to chain fallible operations that may return [`None`].
    ///
    /// ```
    /// let arr_2d = [["A0", "A1"], ["B0", "B1"]];
    ///
    /// let item_0_1 = arr_2d.get(0).and_then(|row| row.get(1));
    /// assert_eq!(item_0_1, Some(&"A1"));
    ///
    /// let item_2_0 = arr_2d.get(2).and_then(|row| row.get(0));
    /// assert_eq!(item_2_0, None);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn and_then<U, F>(self, f: F) -> Option<U>
    where
        F: FnOnce(T) -> Option<U>,
    {
        match self {
            Some(x) => f(x),
            None => None,
        }
    }

    /// Returns [`None`] if the option is [`None`], otherwise calls `predicate`
    /// with the wrapped value and returns:
    ///
    /// - [`Some(t)`] if `predicate` returns `true` (where `t` is the wrapped
    ///   value), and
    /// - [`None`] if `predicate` returns `false`.
    ///
    /// This function works similar to [`Iterator::filter()`]. You can imagine
    /// the `Option<T>` being an iterator over one or zero elements. `filter()`
    /// lets you decide which elements to keep.
    ///
    /// # Examples
    ///
    /// ```rust
    /// fn is_even(n: &i32) -> bool {
    ///     n % 2 == 0
    /// }
    ///
    /// assert_eq!(None.filter(is_even), None);
    /// assert_eq!(Some(3).filter(is_even), None);
    /// assert_eq!(Some(4).filter(is_even), Some(4));
    /// ```
    ///
    /// [`Some(t)`]: Some
    #[inline]
    #[stable(feature = "option_filter", since = "1.27.0")]
    pub fn filter<P>(self, predicate: P) -> Self
    where
        P: FnOnce(&T) -> bool,
    {
        if let Some(x) = self {
            if predicate(&x) {
                return Some(x);
            }
        }
        None
    }

    /// Returns the option if it contains a value, otherwise returns `optb`.
    ///
    /// Arguments passed to `or` are eagerly evaluated; if you are passing the
    /// result of a function call, it is recommended to use [`or_else`], which is
    /// lazily evaluated.
    ///
    /// [`or_else`]: Option::or_else
    ///
    /// # Examples
    ///
    /// ```
    /// let x = Some(2);
    /// let y = None;
    /// assert_eq!(x.or(y), Some(2));
    ///
    /// let x = None;
    /// let y = Some(100);
    /// assert_eq!(x.or(y), Some(100));
    ///
    /// let x = Some(2);
    /// let y = Some(100);
    /// assert_eq!(x.or(y), Some(2));
    ///
    /// let x: Option<u32> = None;
    /// let y = None;
    /// assert_eq!(x.or(y), None);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn or(self, optb: Option<T>) -> Option<T> {
        match self {
            Some(x) => Some(x),
            None => optb,
        }
    }

    /// Returns the option if it contains a value, otherwise calls `f` and
    /// returns the result.
    ///
    /// # Examples
    ///
    /// ```
    /// fn nobody() -> Option<&'static str> { None }
    /// fn vikings() -> Option<&'static str> { Some("vikings") }
    ///
    /// assert_eq!(Some("barbarians").or_else(vikings), Some("barbarians"));
    /// assert_eq!(None.or_else(vikings), Some("vikings"));
    /// assert_eq!(None.or_else(nobody), None);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn or_else<F>(self, f: F) -> Option<T>
    where
        F: FnOnce() -> Option<T>,
    {
        match self {
            Some(x) => Some(x),
            None => f(),
        }
    }

    /// Returns [`Some`] if exactly one of `self`, `optb` is [`Some`], otherwise returns [`None`].
    ///
    /// # Examples
    ///
    /// ```
    /// let x = Some(2);
    /// let y: Option<u32> = None;
    /// assert_eq!(x.xor(y), Some(2));
    ///
    /// let x: Option<u32> = None;
    /// let y = Some(2);
    /// assert_eq!(x.xor(y), Some(2));
    ///
    /// let x = Some(2);
    /// let y = Some(2);
    /// assert_eq!(x.xor(y), None);
    ///
    /// let x: Option<u32> = None;
    /// let y: Option<u32> = None;
    /// assert_eq!(x.xor(y), None);
    /// ```
    #[inline]
    #[stable(feature = "option_xor", since = "1.37.0")]
    pub fn xor(self, optb: Option<T>) -> Option<T> {
        match (self, optb) {
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            _ => None,
        }
    }

    /////////////////////////////////////////////////////////////////////////
    // Entry-like operations to insert a value and return a reference
    /////////////////////////////////////////////////////////////////////////

    /// Inserts `value` into the option, then returns a mutable reference to it.
    ///
    /// If the option already contains a value, the old value is dropped.
    ///
    /// See also [`Option::get_or_insert`], which doesn't update the value if
    /// the option already contains [`Some`].
    ///
    /// # Example
    ///
    /// ```
    /// let mut opt = None;
    /// let val = opt.insert(1);
    /// assert_eq!(*val, 1);
    /// assert_eq!(opt.unwrap(), 1);
    /// let val = opt.insert(2);
    /// assert_eq!(*val, 2);
    /// *val = 3;
    /// assert_eq!(opt.unwrap(), 3);
    /// ```
    #[must_use = "if you intended to set a value, consider assignment instead"]
    #[inline]
    #[stable(feature = "option_insert", since = "1.53.0")]
    pub fn insert(&mut self, value: T) -> &mut T {
        *self = Some(value);

        // SAFETY: the code above just filled the option
        unsafe { self.as_mut().unwrap_unchecked() }
    }

    /// Inserts `value` into the option if it is [`None`], then
    /// returns a mutable reference to the contained value.
    ///
    /// See also [`Option::insert`], which updates the value even if
    /// the option already contains [`Some`].
    ///
    /// # Examples
    ///
    /// ```
    /// let mut x = None;
    ///
    /// {
    ///     let y: &mut u32 = x.get_or_insert(5);
    ///     assert_eq!(y, &5);
    ///
    ///     *y = 7;
    /// }
    ///
    /// assert_eq!(x, Some(7));
    /// ```
    #[inline]
    #[stable(feature = "option_entry", since = "1.20.0")]
    pub fn get_or_insert(&mut self, value: T) -> &mut T {
        if let None = *self {
            *self = Some(value);
        }

        // SAFETY: a `None` variant for `self` would have been replaced by a `Some`
        // variant in the code above.
        unsafe { self.as_mut().unwrap_unchecked() }
    }

    /// Inserts the default value into the option if it is [`None`], then
    /// returns a mutable reference to the contained value.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(option_get_or_insert_default)]
    ///
    /// let mut x = None;
    ///
    /// {
    ///     let y: &mut u32 = x.get_or_insert_default();
    ///     assert_eq!(y, &0);
    ///
    ///     *y = 7;
    /// }
    ///
    /// assert_eq!(x, Some(7));
    /// ```
    #[inline]
    #[unstable(feature = "option_get_or_insert_default", issue = "82901")]
    pub fn get_or_insert_default(&mut self) -> &mut T
    where
        T: Default,
    {
        self.get_or_insert_with(T::default)
    }

    /// Inserts a value computed from `f` into the option if it is [`None`],
    /// then returns a mutable reference to the contained value.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut x = None;
    ///
    /// {
    ///     let y: &mut u32 = x.get_or_insert_with(|| 5);
    ///     assert_eq!(y, &5);
    ///
    ///     *y = 7;
    /// }
    ///
    /// assert_eq!(x, Some(7));
    /// ```
    #[inline]
    #[stable(feature = "option_entry", since = "1.20.0")]
    pub fn get_or_insert_with<F>(&mut self, f: F) -> &mut T
    where
        F: FnOnce() -> T,
    {
        if let None = self {
            *self = Some(f());
        }

        // SAFETY: a `None` variant for `self` would have been replaced by a `Some`
        // variant in the code above.
        unsafe { self.as_mut().unwrap_unchecked() }
    }

    /////////////////////////////////////////////////////////////////////////
    // Misc
    /////////////////////////////////////////////////////////////////////////

    /// Takes the value out of the option, leaving a [`None`] in its place.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut x = Some(2);
    /// let y = x.take();
    /// assert_eq!(x, None);
    /// assert_eq!(y, Some(2));
    ///
    /// let mut x: Option<u32> = None;
    /// let y = x.take();
    /// assert_eq!(x, None);
    /// assert_eq!(y, None);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_unstable(feature = "const_option", issue = "67441")]
    pub const fn take(&mut self) -> Option<T> {
        // FIXME replace `mem::replace` by `mem::take` when the latter is const ready
        mem::replace(self, None)
    }

    /// Replaces the actual value in the option by the value given in parameter,
    /// returning the old value if present,
    /// leaving a [`Some`] in its place without deinitializing either one.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut x = Some(2);
    /// let old = x.replace(5);
    /// assert_eq!(x, Some(5));
    /// assert_eq!(old, Some(2));
    ///
    /// let mut x = None;
    /// let old = x.replace(3);
    /// assert_eq!(x, Some(3));
    /// assert_eq!(old, None);
    /// ```
    #[inline]
    #[rustc_const_unstable(feature = "const_option", issue = "67441")]
    #[stable(feature = "option_replace", since = "1.31.0")]
    pub const fn replace(&mut self, value: T) -> Option<T> {
        mem::replace(self, Some(value))
    }

    /// Zips `self` with another `Option`.
    ///
    /// If `self` is `Some(s)` and `other` is `Some(o)`, this method returns `Some((s, o))`.
    /// Otherwise, `None` is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// let x = Some(1);
    /// let y = Some("hi");
    /// let z = None::<u8>;
    ///
    /// assert_eq!(x.zip(y), Some((1, "hi")));
    /// assert_eq!(x.zip(z), None);
    /// ```
    #[stable(feature = "option_zip_option", since = "1.46.0")]
    pub fn zip<U>(self, other: Option<U>) -> Option<(T, U)> {
        match (self, other) {
            (Some(a), Some(b)) => Some((a, b)),
            _ => None,
        }
    }

    /// Zips `self` and another `Option` with function `f`.
    ///
    /// If `self` is `Some(s)` and `other` is `Some(o)`, this method returns `Some(f(s, o))`.
    /// Otherwise, `None` is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(option_zip)]
    ///
    /// #[derive(Debug, PartialEq)]
    /// struct Point {
    ///     x: f64,
    ///     y: f64,
    /// }
    ///
    /// impl Point {
    ///     fn new(x: f64, y: f64) -> Self {
    ///         Self { x, y }
    ///     }
    /// }
    ///
    /// let x = Some(17.5);
    /// let y = Some(42.7);
    ///
    /// assert_eq!(x.zip_with(y, Point::new), Some(Point { x: 17.5, y: 42.7 }));
    /// assert_eq!(x.zip_with(None, Point::new), None);
    /// ```
    #[unstable(feature = "option_zip", issue = "70086")]
    pub fn zip_with<U, F, R>(self, other: Option<U>, f: F) -> Option<R>
    where
        F: FnOnce(T, U) -> R,
    {
        match (self, other) {
            (Some(a), Some(b)) => Some(f(a, b)),
            _ => None,
        }
    }
}

impl<T, U> Option<(T, U)> {
    /// Unzips an option containing a tuple of two options.
    ///
    /// If `self` is `Some((a, b))` this method returns `(Some(a), Some(b))`.
    /// Otherwise, `(None, None)` is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// let x = Some((1, "hi"));
    /// let y = None::<(u8, u32)>;
    ///
    /// assert_eq!(x.unzip(), (Some(1), Some("hi")));
    /// assert_eq!(y.unzip(), (None, None));
    /// ```
    #[inline]
    #[stable(feature = "unzip_option", since = "1.66.0")]
    pub fn unzip(self) -> (Option<T>, Option<U>) {
        match self {
            Some((a, b)) => (Some(a), Some(b)),
            None => (None, None),
        }
    }
}

impl<T> Option<&T> {
    /// Maps an `Option<&T>` to an `Option<T>` by copying the contents of the
    /// option.
    ///
    /// # Examples
    ///
    /// ```
    /// let x = 12;
    /// let opt_x = Some(&x);
    /// assert_eq!(opt_x, Some(&12));
    /// let copied = opt_x.copied();
    /// assert_eq!(copied, Some(12));
    /// ```
    #[must_use = "`self` will be dropped if the result is not used"]
    #[stable(feature = "copied", since = "1.35.0")]
    #[rustc_const_unstable(feature = "const_option", issue = "67441")]
    pub const fn copied(self) -> Option<T>
    where
        T: Copy,
    {
        // FIXME: this implementation, which sidesteps using `Option::map` since it's not const
        // ready yet, should be reverted when possible to avoid code repetition
        match self {
            Some(&v) => Some(v),
            None => None,
        }
    }

    /// Maps an `Option<&T>` to an `Option<T>` by cloning the contents of the
    /// option.
    ///
    /// # Examples
    ///
    /// ```
    /// let x = 12;
    /// let opt_x = Some(&x);
    /// assert_eq!(opt_x, Some(&12));
    /// let cloned = opt_x.cloned();
    /// assert_eq!(cloned, Some(12));
    /// ```
    #[must_use = "`self` will be dropped if the result is not used"]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn cloned(self) -> Option<T>
    where
        T: Clone,
    {
        match self {
            Some(t) => Some(t.clone()),
            None => None,
        }
    }
}

impl<T> Option<&mut T> {
    /// Maps an `Option<&mut T>` to an `Option<T>` by copying the contents of the
    /// option.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut x = 12;
    /// let opt_x = Some(&mut x);
    /// assert_eq!(opt_x, Some(&mut 12));
    /// let copied = opt_x.copied();
    /// assert_eq!(copied, Some(12));
    /// ```
    #[must_use = "`self` will be dropped if the result is not used"]
    #[stable(feature = "copied", since = "1.35.0")]
    #[rustc_const_unstable(feature = "const_option_ext", issue = "91930")]
    pub const fn copied(self) -> Option<T>
    where
        T: Copy,
    {
        match self {
            Some(&mut t) => Some(t),
            None => None,
        }
    }

    /// Maps an `Option<&mut T>` to an `Option<T>` by cloning the contents of the
    /// option.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut x = 12;
    /// let opt_x = Some(&mut x);
    /// assert_eq!(opt_x, Some(&mut 12));
    /// let cloned = opt_x.cloned();
    /// assert_eq!(cloned, Some(12));
    /// ```
    #[must_use = "`self` will be dropped if the result is not used"]
    #[stable(since = "1.26.0", feature = "option_ref_mut_cloned")]
    pub fn cloned(self) -> Option<T>
    where
        T: Clone,
    {
        match self {
            Some(t) => Some(t.clone()),
            None => None,
        }
    }
}

impl<T, E> Option<Result<T, E>> {
    /// Transposes an `Option` of a [`Result`] into a [`Result`] of an `Option`.
    ///
    /// [`None`] will be mapped to <code>[Ok]\([None])</code>.
    /// <code>[Some]\([Ok]\(\_))</code> and <code>[Some]\([Err]\(\_))</code> will be mapped to
    /// <code>[Ok]\([Some]\(\_))</code> and <code>[Err]\(\_)</code>.
    ///
    /// # Examples
    ///
    /// ```
    /// #[derive(Debug, Eq, PartialEq)]
    /// struct SomeErr;
    ///
    /// let x: Result<Option<i32>, SomeErr> = Ok(Some(5));
    /// let y: Option<Result<i32, SomeErr>> = Some(Ok(5));
    /// assert_eq!(x, y.transpose());
    /// ```
    #[inline]
    #[stable(feature = "transpose_result", since = "1.33.0")]
    #[rustc_const_unstable(feature = "const_option", issue = "67441")]
    pub const fn transpose(self) -> Result<Option<T>, E> {
        match self {
            Some(Ok(x)) => Ok(Some(x)),
            Some(Err(e)) => Err(e),
            None => Ok(None),
        }
    }
}

// This is a separate function to reduce the code size of .expect() itself.
#[cfg_attr(not(feature = "panic_immediate_abort"), inline(never))]
#[cfg_attr(feature = "panic_immediate_abort", inline)]
#[cold]
#[track_caller]
#[rustc_const_unstable(feature = "const_option", issue = "67441")]
const fn expect_failed(msg: &str) -> ! {
    panic_str(msg)
}

/////////////////////////////////////////////////////////////////////////////
// Trait implementations
/////////////////////////////////////////////////////////////////////////////

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> Clone for Option<T>
where
    T: Clone,
{
    #[inline]
    fn clone(&self) -> Self {
        match self {
            Some(x) => Some(x.clone()),
            None => None,
        }
    }

    #[inline]
    fn clone_from(&mut self, source: &Self) {
        match (self, source) {
            (Some(to), Some(from)) => to.clone_from(from),
            (to, from) => *to = from.clone(),
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> Default for Option<T> {
    /// Returns [`None`][Option::None].
    ///
    /// # Examples
    ///
    /// ```
    /// let opt: Option<u32> = Option::default();
    /// assert!(opt.is_none());
    /// ```
    #[inline]
    fn default() -> Option<T> {
        None
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> IntoIterator for Option<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    /// Returns a consuming iterator over the possibly contained value.
    ///
    /// # Examples
    ///
    /// ```
    /// let x = Some("string");
    /// let v: Vec<&str> = x.into_iter().collect();
    /// assert_eq!(v, ["string"]);
    ///
    /// let x = None;
    /// let v: Vec<&str> = x.into_iter().collect();
    /// assert!(v.is_empty());
    /// ```
    #[inline]
    fn into_iter(self) -> IntoIter<T> {
        IntoIter { inner: Item { opt: self } }
    }
}

#[stable(since = "1.4.0", feature = "option_iter")]
impl<'a, T> IntoIterator for &'a Option<T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Iter<'a, T> {
        self.iter()
    }
}

#[stable(since = "1.4.0", feature = "option_iter")]
impl<'a, T> IntoIterator for &'a mut Option<T> {
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;

    fn into_iter(self) -> IterMut<'a, T> {
        self.iter_mut()
    }
}

#[stable(since = "1.12.0", feature = "option_from")]
impl<T> From<T> for Option<T> {
    /// Moves `val` into a new [`Some`].
    ///
    /// # Examples
    ///
    /// ```
    /// let o: Option<u8> = Option::from(67);
    ///
    /// assert_eq!(Some(67), o);
    /// ```
    fn from(val: T) -> Option<T> {
        Some(val)
    }
}

#[stable(feature = "option_ref_from_ref_option", since = "1.30.0")]
impl<'a, T> From<&'a Option<T>> for Option<&'a T> {
    /// Converts from `&Option<T>` to `Option<&T>`.
    ///
    /// # Examples
    ///
    /// Converts an <code>[Option]<[String]></code> into an <code>[Option]<[usize]></code>, preserving
    /// the original. The [`map`] method takes the `self` argument by value, consuming the original,
    /// so this technique uses `from` to first take an [`Option`] to a reference
    /// to the value inside the original.
    ///
    /// [`map`]: Option::map
    /// [String]: ../../std/string/struct.String.html "String"
    ///
    /// ```
    /// let s: Option<String> = Some(String::from("Hello, Rustaceans!"));
    /// let o: Option<usize> = Option::from(&s).map(|ss: &String| ss.len());
    ///
    /// println!("Can still print s: {s:?}");
    ///
    /// assert_eq!(o, Some(18));
    /// ```
    fn from(o: &'a Option<T>) -> Option<&'a T> {
        o.as_ref()
    }
}

#[stable(feature = "option_ref_from_ref_option", since = "1.30.0")]
impl<'a, T> From<&'a mut Option<T>> for Option<&'a mut T> {
    /// Converts from `&mut Option<T>` to `Option<&mut T>`
    ///
    /// # Examples
    ///
    /// ```
    /// let mut s = Some(String::from("Hello"));
    /// let o: Option<&mut String> = Option::from(&mut s);
    ///
    /// match o {
    ///     Some(t) => *t = String::from("Hello, Rustaceans!"),
    ///     None => (),
    /// }
    ///
    /// assert_eq!(s, Some(String::from("Hello, Rustaceans!")));
    /// ```
    fn from(o: &'a mut Option<T>) -> Option<&'a mut T> {
        o.as_mut()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> crate::marker::StructuralPartialEq for Option<T> {}
#[stable(feature = "rust1", since = "1.0.0")]
impl<T: PartialEq> PartialEq for Option<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        SpecOptionPartialEq::eq(self, other)
    }
}

/// This specialization trait is a workaround for LLVM not currently (2023-01)
/// being able to optimize this itself, even though Alive confirms that it would
/// be legal to do so: <https://github.com/llvm/llvm-project/issues/52622>
///
/// Once that's fixed, `Option` should go back to deriving `PartialEq`, as
/// it used to do before <https://github.com/rust-lang/rust/pull/103556>.
#[unstable(feature = "spec_option_partial_eq", issue = "none", reason = "exposed only for rustc")]
#[doc(hidden)]
pub trait SpecOptionPartialEq: Sized {
    fn eq(l: &Option<Self>, other: &Option<Self>) -> bool;
}

#[unstable(feature = "spec_option_partial_eq", issue = "none", reason = "exposed only for rustc")]
impl<T: PartialEq> SpecOptionPartialEq for T {
    #[inline]
    default fn eq(l: &Option<T>, r: &Option<T>) -> bool {
        match (l, r) {
            (Some(l), Some(r)) => *l == *r,
            (None, None) => true,
            _ => false,
        }
    }
}

macro_rules! non_zero_option {
    ( $( #[$stability: meta] $NZ:ty; )+ ) => {
        $(
            #[$stability]
            impl SpecOptionPartialEq for $NZ {
                #[inline]
                fn eq(l: &Option<Self>, r: &Option<Self>) -> bool {
                    l.map(Self::get).unwrap_or(0) == r.map(Self::get).unwrap_or(0)
                }
            }
        )+
    };
}

non_zero_option! {
    #[stable(feature = "nonzero", since = "1.28.0")] crate::num::NonZeroU8;
    #[stable(feature = "nonzero", since = "1.28.0")] crate::num::NonZeroU16;
    #[stable(feature = "nonzero", since = "1.28.0")] crate::num::NonZeroU32;
    #[stable(feature = "nonzero", since = "1.28.0")] crate::num::NonZeroU64;
    #[stable(feature = "nonzero", since = "1.28.0")] crate::num::NonZeroU128;
    #[stable(feature = "nonzero", since = "1.28.0")] crate::num::NonZeroUsize;
    #[stable(feature = "signed_nonzero", since = "1.34.0")] crate::num::NonZeroI8;
    #[stable(feature = "signed_nonzero", since = "1.34.0")] crate::num::NonZeroI16;
    #[stable(feature = "signed_nonzero", since = "1.34.0")] crate::num::NonZeroI32;
    #[stable(feature = "signed_nonzero", since = "1.34.0")] crate::num::NonZeroI64;
    #[stable(feature = "signed_nonzero", since = "1.34.0")] crate::num::NonZeroI128;
    #[stable(feature = "signed_nonzero", since = "1.34.0")] crate::num::NonZeroIsize;
}

#[stable(feature = "nonnull", since = "1.25.0")]
impl<T> SpecOptionPartialEq for crate::ptr::NonNull<T> {
    #[inline]
    fn eq(l: &Option<Self>, r: &Option<Self>) -> bool {
        l.map(Self::as_ptr).unwrap_or_else(|| crate::ptr::null_mut())
            == r.map(Self::as_ptr).unwrap_or_else(|| crate::ptr::null_mut())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl SpecOptionPartialEq for cmp::Ordering {
    #[inline]
    fn eq(l: &Option<Self>, r: &Option<Self>) -> bool {
        l.map_or(2, |x| x as i8) == r.map_or(2, |x| x as i8)
    }
}

/////////////////////////////////////////////////////////////////////////////
// The Option Iterators
/////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Debug)]
struct Item<A> {
    opt: Option<A>,
}

impl<A> Iterator for Item<A> {
    type Item = A;

    #[inline]
    fn next(&mut self) -> Option<A> {
        self.opt.take()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self.opt {
            Some(_) => (1, Some(1)),
            None => (0, Some(0)),
        }
    }
}

impl<A> DoubleEndedIterator for Item<A> {
    #[inline]
    fn next_back(&mut self) -> Option<A> {
        self.opt.take()
    }
}

impl<A> ExactSizeIterator for Item<A> {}
impl<A> FusedIterator for Item<A> {}
unsafe impl<A> TrustedLen for Item<A> {}

/// An iterator over a reference to the [`Some`] variant of an [`Option`].
///
/// The iterator yields one value if the [`Option`] is a [`Some`], otherwise none.
///
/// This `struct` is created by the [`Option::iter`] function.
#[stable(feature = "rust1", since = "1.0.0")]
#[derive(Debug)]
pub struct Iter<'a, A: 'a> {
    inner: Item<&'a A>,
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, A> Iterator for Iter<'a, A> {
    type Item = &'a A;

    #[inline]
    fn next(&mut self) -> Option<&'a A> {
        self.inner.next()
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, A> DoubleEndedIterator for Iter<'a, A> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a A> {
        self.inner.next_back()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A> ExactSizeIterator for Iter<'_, A> {}

#[stable(feature = "fused", since = "1.26.0")]
impl<A> FusedIterator for Iter<'_, A> {}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<A> TrustedLen for Iter<'_, A> {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A> Clone for Iter<'_, A> {
    #[inline]
    fn clone(&self) -> Self {
        Iter { inner: self.inner.clone() }
    }
}

/// An iterator over a mutable reference to the [`Some`] variant of an [`Option`].
///
/// The iterator yields one value if the [`Option`] is a [`Some`], otherwise none.
///
/// This `struct` is created by the [`Option::iter_mut`] function.
#[stable(feature = "rust1", since = "1.0.0")]
#[derive(Debug)]
pub struct IterMut<'a, A: 'a> {
    inner: Item<&'a mut A>,
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, A> Iterator for IterMut<'a, A> {
    type Item = &'a mut A;

    #[inline]
    fn next(&mut self) -> Option<&'a mut A> {
        self.inner.next()
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, A> DoubleEndedIterator for IterMut<'a, A> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a mut A> {
        self.inner.next_back()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A> ExactSizeIterator for IterMut<'_, A> {}

#[stable(feature = "fused", since = "1.26.0")]
impl<A> FusedIterator for IterMut<'_, A> {}
#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<A> TrustedLen for IterMut<'_, A> {}

/// An iterator over the value in [`Some`] variant of an [`Option`].
///
/// The iterator yields one value if the [`Option`] is a [`Some`], otherwise none.
///
/// This `struct` is created by the [`Option::into_iter`] function.
#[derive(Clone, Debug)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct IntoIter<A> {
    inner: Item<A>,
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A> Iterator for IntoIter<A> {
    type Item = A;

    #[inline]
    fn next(&mut self) -> Option<A> {
        self.inner.next()
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A> DoubleEndedIterator for IntoIter<A> {
    #[inline]
    fn next_back(&mut self) -> Option<A> {
        self.inner.next_back()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A> ExactSizeIterator for IntoIter<A> {}

#[stable(feature = "fused", since = "1.26.0")]
impl<A> FusedIterator for IntoIter<A> {}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<A> TrustedLen for IntoIter<A> {}

/////////////////////////////////////////////////////////////////////////////
// FromIterator
/////////////////////////////////////////////////////////////////////////////

#[stable(feature = "rust1", since = "1.0.0")]
impl<A, V: FromIterator<A>> FromIterator<Option<A>> for Option<V> {
    /// Takes each element in the [`Iterator`]: if it is [`None`][Option::None],
    /// no further elements are taken, and the [`None`][Option::None] is
    /// returned. Should no [`None`][Option::None] occur, a container of type
    /// `V` containing the values of each [`Option`] is returned.
    ///
    /// # Examples
    ///
    /// Here is an example which increments every integer in a vector.
    /// We use the checked variant of `add` that returns `None` when the
    /// calculation would result in an overflow.
    ///
    /// ```
    /// let items = vec![0_u16, 1, 2];
    ///
    /// let res: Option<Vec<u16>> = items
    ///     .iter()
    ///     .map(|x| x.checked_add(1))
    ///     .collect();
    ///
    /// assert_eq!(res, Some(vec![1, 2, 3]));
    /// ```
    ///
    /// As you can see, this will return the expected, valid items.
    ///
    /// Here is another example that tries to subtract one from another list
    /// of integers, this time checking for underflow:
    ///
    /// ```
    /// let items = vec![2_u16, 1, 0];
    ///
    /// let res: Option<Vec<u16>> = items
    ///     .iter()
    ///     .map(|x| x.checked_sub(1))
    ///     .collect();
    ///
    /// assert_eq!(res, None);
    /// ```
    ///
    /// Since the last element is zero, it would underflow. Thus, the resulting
    /// value is `None`.
    ///
    /// Here is a variation on the previous example, showing that no
    /// further elements are taken from `iter` after the first `None`.
    ///
    /// ```
    /// let items = vec![3_u16, 2, 1, 10];
    ///
    /// let mut shared = 0;
    ///
    /// let res: Option<Vec<u16>> = items
    ///     .iter()
    ///     .map(|x| { shared += x; x.checked_sub(2) })
    ///     .collect();
    ///
    /// assert_eq!(res, None);
    /// assert_eq!(shared, 6);
    /// ```
    ///
    /// Since the third element caused an underflow, no further elements were taken,
    /// so the final value of `shared` is 6 (= `3 + 2 + 1`), not 16.
    #[inline]
    fn from_iter<I: IntoIterator<Item = Option<A>>>(iter: I) -> Option<V> {
        // FIXME(#11084): This could be replaced with Iterator::scan when this
        // performance bug is closed.

        iter::try_process(iter.into_iter(), |i| i.collect())
    }
}

#[unstable(feature = "try_trait_v2", issue = "84277")]
impl<T> ops::Try for Option<T> {
    type Output = T;
    type Residual = Option<convert::Infallible>;

    #[inline]
    fn from_output(output: Self::Output) -> Self {
        Some(output)
    }

    #[inline]
    fn branch(self) -> ControlFlow<Self::Residual, Self::Output> {
        match self {
            Some(v) => ControlFlow::Continue(v),
            None => ControlFlow::Break(None),
        }
    }
}

#[unstable(feature = "try_trait_v2", issue = "84277")]
impl<T> ops::FromResidual for Option<T> {
    #[inline]
    fn from_residual(residual: Option<convert::Infallible>) -> Self {
        match residual {
            None => None,
        }
    }
}

#[unstable(feature = "try_trait_v2_yeet", issue = "96374")]
impl<T> ops::FromResidual<ops::Yeet<()>> for Option<T> {
    #[inline]
    fn from_residual(ops::Yeet(()): ops::Yeet<()>) -> Self {
        None
    }
}

#[unstable(feature = "try_trait_v2_residual", issue = "91285")]
impl<T> ops::Residual<T> for Option<convert::Infallible> {
    type TryType = Option<T>;
}

impl<T> Option<Option<T>> {
    /// Converts from `Option<Option<T>>` to `Option<T>`.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let x: Option<Option<u32>> = Some(Some(6));
    /// assert_eq!(Some(6), x.flatten());
    ///
    /// let x: Option<Option<u32>> = Some(None);
    /// assert_eq!(None, x.flatten());
    ///
    /// let x: Option<Option<u32>> = None;
    /// assert_eq!(None, x.flatten());
    /// ```
    ///
    /// Flattening only removes one level of nesting at a time:
    ///
    /// ```
    /// let x: Option<Option<Option<u32>>> = Some(Some(Some(6)));
    /// assert_eq!(Some(Some(6)), x.flatten());
    /// assert_eq!(Some(6), x.flatten().flatten());
    /// ```
    #[inline]
    #[stable(feature = "option_flattening", since = "1.40.0")]
    #[rustc_const_unstable(feature = "const_option", issue = "67441")]
    pub const fn flatten(self) -> Option<T> {
        match self {
            Some(inner) => inner,
            None => None,
        }
    }
}
