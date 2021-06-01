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
//!     Some(x) => println!("Result: {}", x),
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
//! the optional owned box, [`Option`]`<`[`Box<T>`]`>`.
//!
//! The following example uses [`Option`] to create an optional box of
//! [`i32`]. Notice that in order to use the inner [`i32`] value first, the
//! `check_optional` function needs to use pattern matching to
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
//!         Some(p) => println!("has value {}", p),
//!         None => println!("has no value"),
//!     }
//! }
//! ```
//!
//! # Representation
//!
//! Rust guarantees to optimize the following types `T` such that
//! [`Option<T>`] has the same size as `T`:
//!
//! * [`Box<U>`]
//! * `&U`
//! * `&mut U`
//! * `fn`, `extern "C" fn`
//! * [`num::NonZero*`]
//! * [`ptr::NonNull<U>`]
//! * `#[repr(transparent)]` struct around one of the types in this list.
//!
//! This is called the "null pointer optimization" or NPO.
//!
//! It is further guaranteed that, for the cases above, one can
//! [`mem::transmute`] from all valid values of `T` to `Option<T>` and
//! from `Some::<T>(_)` to `T` (but transmuting `None::<T>` to `T`
//! is undefined behaviour).
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
//!     Some(name) => println!("the biggest animal is {}", name),
//!     None => println!("there are no animals :("),
//! }
//! ```
//!
//! [`Box<T>`]: ../../std/boxed/struct.Box.html
//! [`Box<U>`]: ../../std/boxed/struct.Box.html
//! [`num::NonZero*`]: crate::num
//! [`ptr::NonNull<U>`]: crate::ptr::NonNull

#![stable(feature = "rust1", since = "1.0.0")]

use crate::iter::{FromIterator, FusedIterator, TrustedLen};
use crate::pin::Pin;
use crate::{
    convert, hint, mem,
    ops::{self, ControlFlow, Deref, DerefMut},
};

/// The `Option` type. See [the module level documentation](self) for more.
#[derive(Copy, PartialEq, PartialOrd, Eq, Ord, Debug, Hash)]
#[rustc_diagnostic_item = "option_type"]
#[stable(feature = "rust1", since = "1.0.0")]
pub enum Option<T> {
    /// No value
    #[lang = "None"]
    #[stable(feature = "rust1", since = "1.0.0")]
    None,
    /// Some value `T`
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
    #[rustc_const_stable(feature = "const_option", since = "1.48.0")]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const fn is_some(&self) -> bool {
        matches!(*self, Some(_))
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
    #[rustc_const_stable(feature = "const_option", since = "1.48.0")]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub const fn is_none(&self) -> bool {
        !self.is_some()
    }

    /// Returns `true` if the option is a [`Some`] value containing the given value.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(option_result_contains)]
    ///
    /// let x: Option<u32> = Some(2);
    /// assert_eq!(x.contains(&2), true);
    ///
    /// let x: Option<u32> = Some(3);
    /// assert_eq!(x.contains(&2), false);
    ///
    /// let x: Option<u32> = None;
    /// assert_eq!(x.contains(&2), false);
    /// ```
    #[must_use]
    #[inline]
    #[unstable(feature = "option_result_contains", issue = "62358")]
    pub fn contains<U>(&self, x: &U) -> bool
    where
        U: PartialEq<T>,
    {
        match self {
            Some(y) => x == y,
            None => false,
        }
    }

    /////////////////////////////////////////////////////////////////////////
    // Adapter for working with references
    /////////////////////////////////////////////////////////////////////////

    /// Converts from `&Option<T>` to `Option<&T>`.
    ///
    /// # Examples
    ///
    /// Converts an `Option<`[`String`]`>` into an `Option<`[`usize`]`>`, preserving the original.
    /// The [`map`] method takes the `self` argument by value, consuming the original,
    /// so this technique uses `as_ref` to first take an `Option` to a reference
    /// to the value inside the original.
    ///
    /// [`map`]: Option::map
    /// [`String`]: ../../std/string/struct.String.html
    ///
    /// ```
    /// let text: Option<String> = Some("Hello, world!".to_string());
    /// // First, cast `Option<String>` to `Option<&String>` with `as_ref`,
    /// // then consume *that* with `map`, leaving `text` on the stack.
    /// let text_length: Option<usize> = text.as_ref().map(|s| s.len());
    /// println!("still can print text: {:?}", text);
    /// ```
    #[inline]
    #[rustc_const_stable(feature = "const_option", since = "1.48.0")]
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
    pub fn as_mut(&mut self) -> Option<&mut T> {
        match *self {
            Some(ref mut x) => Some(x),
            None => None,
        }
    }

    /// Converts from [`Pin`]`<&Option<T>>` to `Option<`[`Pin`]`<&T>>`.
    #[inline]
    #[stable(feature = "pin", since = "1.33.0")]
    pub fn as_pin_ref(self: Pin<&Self>) -> Option<Pin<&T>> {
        // SAFETY: `x` is guaranteed to be pinned because it comes from `self`
        // which is pinned.
        unsafe { Pin::get_ref(self).as_ref().map(|x| Pin::new_unchecked(x)) }
    }

    /// Converts from [`Pin`]`<&mut Option<T>>` to `Option<`[`Pin`]`<&mut T>>`.
    #[inline]
    #[stable(feature = "pin", since = "1.33.0")]
    pub fn as_pin_mut(self: Pin<&mut Self>) -> Option<Pin<&mut T>> {
        // SAFETY: `get_unchecked_mut` is never used to move the `Option` inside `self`.
        // `x` is guaranteed to be pinned because it comes from `self` which is pinned.
        unsafe { Pin::get_unchecked_mut(self).as_mut().map(|x| Pin::new_unchecked(x)) }
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
    #[inline]
    #[track_caller]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn expect(self, msg: &str) -> T {
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
            None => panic!("called `Option::unwrap()` on a `None` value"),
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
    pub fn unwrap_or_else<F: FnOnce() -> T>(self, f: F) -> T {
        match self {
            Some(x) => x,
            None => f(),
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
    /// #![feature(option_result_unwrap_unchecked)]
    /// let x = Some("air");
    /// assert_eq!(unsafe { x.unwrap_unchecked() }, "air");
    /// ```
    ///
    /// ```no_run
    /// #![feature(option_result_unwrap_unchecked)]
    /// let x: Option<&str> = None;
    /// assert_eq!(unsafe { x.unwrap_unchecked() }, "air"); // Undefined behavior!
    /// ```
    #[inline]
    #[track_caller]
    #[unstable(feature = "option_result_unwrap_unchecked", reason = "newly added", issue = "81383")]
    pub unsafe fn unwrap_unchecked(self) -> T {
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

    /// Maps an `Option<T>` to `Option<U>` by applying a function to a contained value.
    ///
    /// # Examples
    ///
    /// Converts an `Option<`[`String`]`>` into an `Option<`[`usize`]`>`, consuming the original:
    ///
    /// [`String`]: ../../std/string/struct.String.html
    /// ```
    /// let maybe_some_string = Some(String::from("Hello, World!"));
    /// // `Option::map` takes self *by value*, consuming `maybe_some_string`
    /// let maybe_some_len = maybe_some_string.map(|s| s.len());
    ///
    /// assert_eq!(maybe_some_len, Some(13));
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn map<U, F: FnOnce(T) -> U>(self, f: F) -> Option<U> {
        match self {
            Some(x) => Some(f(x)),
            None => None,
        }
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
    pub fn map_or<U, F: FnOnce(T) -> U>(self, default: U, f: F) -> U {
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
    pub fn map_or_else<U, D: FnOnce() -> U, F: FnOnce(T) -> U>(self, default: D, f: F) -> U {
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
    pub fn ok_or_else<E, F: FnOnce() -> E>(self, err: F) -> Result<T, E> {
        match self {
            Some(v) => Ok(v),
            None => Err(err()),
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
    /// fn sq(x: u32) -> Option<u32> { Some(x * x) }
    /// fn nope(_: u32) -> Option<u32> { None }
    ///
    /// assert_eq!(Some(2).and_then(sq).and_then(sq), Some(16));
    /// assert_eq!(Some(2).and_then(sq).and_then(nope), None);
    /// assert_eq!(Some(2).and_then(nope).and_then(sq), None);
    /// assert_eq!(None.and_then(sq).and_then(sq), None);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn and_then<U, F: FnOnce(T) -> Option<U>>(self, f: F) -> Option<U> {
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
    pub fn filter<P: FnOnce(&T) -> bool>(self, predicate: P) -> Self {
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
            Some(_) => self,
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
    pub fn or_else<F: FnOnce() -> Option<T>>(self, f: F) -> Option<T> {
        match self {
            Some(_) => self,
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

    /// Inserts `value` into the option then returns a mutable reference to it.
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
    #[inline]
    #[stable(feature = "option_insert", since = "1.53.0")]
    pub fn insert(&mut self, value: T) -> &mut T {
        *self = Some(value);

        match self {
            Some(v) => v,
            // SAFETY: the code above just filled the option
            None => unsafe { hint::unreachable_unchecked() },
        }
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
        self.get_or_insert_with(|| value)
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
        self.get_or_insert_with(Default::default)
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
    pub fn get_or_insert_with<F: FnOnce() -> T>(&mut self, f: F) -> &mut T {
        if let None = *self {
            *self = Some(f());
        }

        match self {
            Some(v) => v,
            // SAFETY: a `None` variant for `self` would have been replaced by a `Some`
            // variant in the code above.
            None => unsafe { hint::unreachable_unchecked() },
        }
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
    pub fn take(&mut self) -> Option<T> {
        mem::take(self)
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
    #[stable(feature = "option_replace", since = "1.31.0")]
    pub fn replace(&mut self, value: T) -> Option<T> {
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
        Some(f(self?, other?))
    }
}

impl<T: Copy> Option<&T> {
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
    #[stable(feature = "copied", since = "1.35.0")]
    pub fn copied(self) -> Option<T> {
        self.map(|&t| t)
    }
}

impl<T: Copy> Option<&mut T> {
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
    #[stable(feature = "copied", since = "1.35.0")]
    pub fn copied(self) -> Option<T> {
        self.map(|&mut t| t)
    }
}

impl<T: Clone> Option<&T> {
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
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn cloned(self) -> Option<T> {
        self.map(|t| t.clone())
    }
}

impl<T: Clone> Option<&mut T> {
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
    #[stable(since = "1.26.0", feature = "option_ref_mut_cloned")]
    pub fn cloned(self) -> Option<T> {
        self.map(|t| t.clone())
    }
}

impl<T: Default> Option<T> {
    /// Returns the contained [`Some`] value or a default
    ///
    /// Consumes the `self` argument then, if [`Some`], returns the contained
    /// value, otherwise if [`None`], returns the [default value] for that
    /// type.
    ///
    /// # Examples
    ///
    /// Converts a string to an integer, turning poorly-formed strings
    /// into 0 (the default value for integers). [`parse`] converts
    /// a string to any other type that implements [`FromStr`], returning
    /// [`None`] on error.
    ///
    /// ```
    /// let good_year_from_input = "1909";
    /// let bad_year_from_input = "190blarg";
    /// let good_year = good_year_from_input.parse().ok().unwrap_or_default();
    /// let bad_year = bad_year_from_input.parse().ok().unwrap_or_default();
    ///
    /// assert_eq!(1909, good_year);
    /// assert_eq!(0, bad_year);
    /// ```
    ///
    /// [default value]: Default::default
    /// [`parse`]: str::parse
    /// [`FromStr`]: crate::str::FromStr
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn unwrap_or_default(self) -> T {
        match self {
            Some(x) => x,
            None => Default::default(),
        }
    }
}

impl<T: Deref> Option<T> {
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
    #[stable(feature = "option_deref", since = "1.40.0")]
    pub fn as_deref(&self) -> Option<&T::Target> {
        self.as_ref().map(|t| t.deref())
    }
}

impl<T: DerefMut> Option<T> {
    /// Converts from `Option<T>` (or `&mut Option<T>`) to `Option<&mut T::Target>`.
    ///
    /// Leaves the original `Option` in-place, creating a new one containing a mutable reference to
    /// the inner type's `Deref::Target` type.
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
    #[stable(feature = "option_deref", since = "1.40.0")]
    pub fn as_deref_mut(&mut self) -> Option<&mut T::Target> {
        self.as_mut().map(|t| t.deref_mut())
    }
}

impl<T, E> Option<Result<T, E>> {
    /// Transposes an `Option` of a [`Result`] into a [`Result`] of an `Option`.
    ///
    /// [`None`] will be mapped to [`Ok`]`(`[`None`]`)`.
    /// [`Some`]`(`[`Ok`]`(_))` and [`Some`]`(`[`Err`]`(_))` will be mapped to
    /// [`Ok`]`(`[`Some`]`(_))` and [`Err`]`(_)`.
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
#[inline(never)]
#[cold]
#[track_caller]
fn expect_failed(msg: &str) -> ! {
    panic!("{}", msg)
}

/////////////////////////////////////////////////////////////////////////////
// Trait implementations
/////////////////////////////////////////////////////////////////////////////

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Clone> Clone for Option<T> {
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
    /// Copies `val` into a new `Some`.
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
    /// Converts an `Option<`[`String`]`>` into an `Option<`[`usize`]`>`, preserving the original.
    /// The [`map`] method takes the `self` argument by value, consuming the original,
    /// so this technique uses `as_ref` to first take an `Option` to a reference
    /// to the value inside the original.
    ///
    /// [`map`]: Option::map
    /// [`String`]: ../../std/string/struct.String.html
    ///
    /// ```
    /// let s: Option<String> = Some(String::from("Hello, Rustaceans!"));
    /// let o: Option<usize> = Option::from(&s).map(|ss: &String| ss.len());
    ///
    /// println!("Can still print s: {:?}", s);
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
    /// returned. Should no [`None`][Option::None] occur, a container with the
    /// values of each [`Option`] is returned.
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

        iter.into_iter().map(|x| x.ok_or(())).collect::<Result<_, _>>().ok()
    }
}

/// The error type that results from applying the try operator (`?`) to a `None` value. If you wish
/// to allow `x?` (where `x` is an `Option<T>`) to be converted into your error type, you can
/// implement `impl From<NoneError>` for `YourErrorType`. In that case, `x?` within a function that
/// returns `Result<_, YourErrorType>` will translate a `None` value into an `Err` result.
#[rustc_diagnostic_item = "none_error"]
#[unstable(feature = "try_trait", issue = "42327")]
#[derive(Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Debug, Hash)]
#[cfg(bootstrap)]
pub struct NoneError;

#[unstable(feature = "try_trait", issue = "42327")]
#[cfg(bootstrap)]
impl<T> ops::TryV1 for Option<T> {
    type Output = T;
    type Error = NoneError;

    #[inline]
    fn into_result(self) -> Result<T, NoneError> {
        self.ok_or(NoneError)
    }

    #[inline]
    fn from_ok(v: T) -> Self {
        Some(v)
    }

    #[inline]
    fn from_error(_: NoneError) -> Self {
        None
    }
}

#[unstable(feature = "try_trait_v2", issue = "84277")]
impl<T> ops::TryV2 for Option<T> {
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

impl<T> Option<Option<T>> {
    /// Converts from `Option<Option<T>>` to `Option<T>`
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
