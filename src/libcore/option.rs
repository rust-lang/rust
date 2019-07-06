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
//! * Return value for otherwise reporting simple errors, where `None` is
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
//! no "null" pointers. Instead, Rust has *optional* pointers, like
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
//!         Some(ref p) => println!("has value {}", p),
//!         None => println!("has no value"),
//!     }
//! }
//! ```
//!
//! This usage of [`Option`] to create safe nullable pointers is so
//! common that Rust does special optimizations to make the
//! representation of [`Option`]`<`[`Box<T>`]`>` a single pointer. Optional pointers
//! in Rust are stored as efficiently as any other pointer type.
//!
//! # Examples
//!
//! Basic pattern matching on [`Option`]:
//!
//! ```
//! let msg = Some("howdy");
//!
//! // Take a reference to the contained string
//! if let Some(ref m) = msg {
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
//! [`Option`]: enum.Option.html
//! [`Some`]: enum.Option.html#variant.Some
//! [`None`]: enum.Option.html#variant.None
//! [`Box<T>`]: ../../std/boxed/struct.Box.html
//! [`i32`]: ../../std/primitive.i32.html

#![stable(feature = "rust1", since = "1.0.0")]

use crate::iter::{FromIterator, FusedIterator, TrustedLen};
use crate::{convert, hint, mem, ops::{self, Deref}};
use crate::pin::Pin;

// Note that this is not a lang item per se, but it has a hidden dependency on
// `Iterator`, which is one. The compiler assumes that the `next` method of
// `Iterator` is an enumeration with one type parameter and two variants,
// which basically means it must be `Option`.

/// The `Option` type. See [the module level documentation](index.html) for more.
#[derive(Copy, PartialEq, PartialOrd, Eq, Ord, Debug, Hash)]
#[stable(feature = "rust1", since = "1.0.0")]
pub enum Option<T> {
    /// No value
    #[stable(feature = "rust1", since = "1.0.0")]
    None,
    /// Some value `T`
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
    ///
    /// [`Some`]: #variant.Some
    #[must_use]
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn is_some(&self) -> bool {
        match *self {
            Some(_) => true,
            None => false,
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
    ///
    /// [`None`]: #variant.None
    #[must_use]
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn is_none(&self) -> bool {
        !self.is_some()
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
    /// [`map`]: enum.Option.html#method.map
    /// [`String`]: ../../std/string/struct.String.html
    /// [`usize`]: ../../std/primitive.usize.html
    ///
    /// ```
    /// let text: Option<String> = Some("Hello, world!".to_string());
    /// // First, cast `Option<String>` to `Option<&String>` with `as_ref`,
    /// // then consume *that* with `map`, leaving `text` on the stack.
    /// let text_length: Option<usize> = text.as_ref().map(|s| s.len());
    /// println!("still can print text: {:?}", text);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn as_ref(&self) -> Option<&T> {
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


    /// Converts from `Pin<&Option<T>>` to `Option<Pin<&T>>`
    #[inline]
    #[stable(feature = "pin", since = "1.33.0")]
    pub fn as_pin_ref<'a>(self: Pin<&'a Option<T>>) -> Option<Pin<&'a T>> {
        unsafe {
            Pin::get_ref(self).as_ref().map(|x| Pin::new_unchecked(x))
        }
    }

    /// Converts from `Pin<&mut Option<T>>` to `Option<Pin<&mut T>>`
    #[inline]
    #[stable(feature = "pin", since = "1.33.0")]
    pub fn as_pin_mut<'a>(self: Pin<&'a mut Option<T>>) -> Option<Pin<&'a mut T>> {
        unsafe {
            Pin::get_unchecked_mut(self).as_mut().map(|x| Pin::new_unchecked(x))
        }
    }

    /////////////////////////////////////////////////////////////////////////
    // Getting to contained values
    /////////////////////////////////////////////////////////////////////////

    /// Unwraps an option, yielding the content of a [`Some`].
    ///
    /// # Panics
    ///
    /// Panics if the value is a [`None`] with a custom panic message provided by
    /// `msg`.
    ///
    /// [`Some`]: #variant.Some
    /// [`None`]: #variant.None
    ///
    /// # Examples
    ///
    /// ```
    /// let x = Some("value");
    /// assert_eq!(x.expect("the world is ending"), "value");
    /// ```
    ///
    /// ```{.should_panic}
    /// let x: Option<&str> = None;
    /// x.expect("the world is ending"); // panics with `the world is ending`
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn expect(self, msg: &str) -> T {
        match self {
            Some(val) => val,
            None => expect_failed(msg),
        }
    }

    /// Moves the value `v` out of the `Option<T>` if it is [`Some(v)`].
    ///
    /// In general, because this function may panic, its use is discouraged.
    /// Instead, prefer to use pattern matching and handle the [`None`]
    /// case explicitly.
    ///
    /// # Panics
    ///
    /// Panics if the self value equals [`None`].
    ///
    /// [`Some(v)`]: #variant.Some
    /// [`None`]: #variant.None
    ///
    /// # Examples
    ///
    /// ```
    /// let x = Some("air");
    /// assert_eq!(x.unwrap(), "air");
    /// ```
    ///
    /// ```{.should_panic}
    /// let x: Option<&str> = None;
    /// assert_eq!(x.unwrap(), "air"); // fails
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn unwrap(self) -> T {
        match self {
            Some(val) => val,
            None => panic!("called `Option::unwrap()` on a `None` value"),
        }
    }

    /// Returns the contained value or a default.
    ///
    /// Arguments passed to `unwrap_or` are eagerly evaluated; if you are passing
    /// the result of a function call, it is recommended to use [`unwrap_or_else`],
    /// which is lazily evaluated.
    ///
    /// [`unwrap_or_else`]: #method.unwrap_or_else
    ///
    /// # Examples
    ///
    /// ```
    /// assert_eq!(Some("car").unwrap_or("bike"), "car");
    /// assert_eq!(None.unwrap_or("bike"), "bike");
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn unwrap_or(self, def: T) -> T {
        match self {
            Some(x) => x,
            None => def,
        }
    }

    /// Returns the contained value or computes it from a closure.
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
    /// [`usize`]: ../../std/primitive.usize.html
    ///
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

    /// Applies a function to the contained value (if any),
    /// or returns the provided default (if not).
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

    /// Applies a function to the contained value (if any),
    /// or computes a default (if not).
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
    /// [`Result<T, E>`]: ../../std/result/enum.Result.html
    /// [`Ok(v)`]: ../../std/result/enum.Result.html#variant.Ok
    /// [`Err(err)`]: ../../std/result/enum.Result.html#variant.Err
    /// [`None`]: #variant.None
    /// [`Some(v)`]: #variant.Some
    /// [`ok_or_else`]: #method.ok_or_else
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
    /// [`Result<T, E>`]: ../../std/result/enum.Result.html
    /// [`Ok(v)`]: ../../std/result/enum.Result.html#variant.Ok
    /// [`Err(err())`]: ../../std/result/enum.Result.html#variant.Err
    /// [`None`]: #variant.None
    /// [`Some(v)`]: #variant.Some
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
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn iter(&self) -> Iter<'_, T> {
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
    /// [`None`]: #variant.None
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
    /// [`None`]: #variant.None
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

    /// Returns `None` if the option is `None`, otherwise calls `predicate`
    /// with the wrapped value and returns:
    ///
    /// - `Some(t)` if `predicate` returns `true` (where `t` is the wrapped
    ///   value), and
    /// - `None` if `predicate` returns `false`.
    ///
    /// This function works similar to `Iterator::filter()`. You can imagine
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
    #[inline]
    #[stable(feature = "option_filter", since = "1.27.0")]
    pub fn filter<P: FnOnce(&T) -> bool>(self, predicate: P) -> Self {
        if let Some(x) = self {
            if predicate(&x) {
                return Some(x)
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
    /// [`or_else`]: #method.or_else
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

    /// Returns [`Some`] if exactly one of `self`, `optb` is [`Some`], otherwise returns `None`.
    ///
    /// [`Some`]: #variant.Some
    /// [`None`]: #variant.None
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
    // Entry-like operations to insert if None and return a reference
    /////////////////////////////////////////////////////////////////////////

    /// Inserts `v` into the option if it is [`None`], then
    /// returns a mutable reference to the contained value.
    ///
    /// [`None`]: #variant.None
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
    pub fn get_or_insert(&mut self, v: T) -> &mut T {
        self.get_or_insert_with(|| v)
    }

    /// Inserts a value computed from `f` into the option if it is [`None`], then
    /// returns a mutable reference to the contained value.
    ///
    /// [`None`]: #variant.None
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
        match *self {
            None => *self = Some(f()),
            _ => (),
        }

        match *self {
            Some(ref mut v) => v,
            None => unsafe { hint::unreachable_unchecked() },
        }
    }

    /////////////////////////////////////////////////////////////////////////
    // Misc
    /////////////////////////////////////////////////////////////////////////

    /// Takes the value out of the option, leaving a [`None`] in its place.
    ///
    /// [`None`]: #variant.None
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
    /// [`Some`]: #variant.Some
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
    /// Returns the contained value or a default
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
    /// [`Some`]: #variant.Some
    /// [`None`]: #variant.None
    /// [default value]: ../default/trait.Default.html#tymethod.default
    /// [`parse`]: ../../std/primitive.str.html#method.parse
    /// [`FromStr`]: ../../std/str/trait.FromStr.html
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn unwrap_or_default(self) -> T {
        match self {
            Some(x) => x,
            None => Default::default(),
        }
    }
}

#[unstable(feature = "inner_deref", reason = "newly added", issue = "50264")]
impl<T: Deref> Option<T> {
    /// Converts from `&Option<T>` to `Option<&T::Target>`.
    ///
    /// Leaves the original Option in-place, creating a new one with a reference
    /// to the original one, additionally coercing the contents via `Deref`.
    pub fn deref(&self) -> Option<&T::Target> {
        self.as_ref().map(|t| t.deref())
    }
}

impl<T, E> Option<Result<T, E>> {
    /// Transposes an `Option` of a `Result` into a `Result` of an `Option`.
    ///
    /// `None` will be mapped to `Ok(None)`.
    /// `Some(Ok(_))` and `Some(Err(_))` will be mapped to `Ok(Some(_))` and `Err(_)`.
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
    pub fn transpose(self) -> Result<Option<T>, E> {
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
    #[inline]
    fn default() -> Option<T> { None }
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
    fn from(val: T) -> Option<T> {
        Some(val)
    }
}

#[stable(feature = "option_ref_from_ref_option", since = "1.30.0")]
impl<'a, T> From<&'a Option<T>> for Option<&'a T> {
    fn from(o: &'a Option<T>) -> Option<&'a T> {
        o.as_ref()
    }
}

#[stable(feature = "option_ref_from_ref_option", since = "1.30.0")]
impl<'a, T> From<&'a mut Option<T>> for Option<&'a mut T> {
    fn from(o: &'a mut Option<T>) -> Option<&'a mut T> {
        o.as_mut()
    }
}

/////////////////////////////////////////////////////////////////////////////
// The Option Iterators
/////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Debug)]
struct Item<A> {
    opt: Option<A>
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
///
/// [`Option`]: enum.Option.html
/// [`Some`]: enum.Option.html#variant.Some
/// [`Option::iter`]: enum.Option.html#method.iter
#[stable(feature = "rust1", since = "1.0.0")]
#[derive(Debug)]
pub struct Iter<'a, A: 'a> { inner: Item<&'a A> }

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, A> Iterator for Iter<'a, A> {
    type Item = &'a A;

    #[inline]
    fn next(&mut self) -> Option<&'a A> { self.inner.next() }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) { self.inner.size_hint() }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, A> DoubleEndedIterator for Iter<'a, A> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a A> { self.inner.next_back() }
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
///
/// [`Option`]: enum.Option.html
/// [`Some`]: enum.Option.html#variant.Some
/// [`Option::iter_mut`]: enum.Option.html#method.iter_mut
#[stable(feature = "rust1", since = "1.0.0")]
#[derive(Debug)]
pub struct IterMut<'a, A: 'a> { inner: Item<&'a mut A> }

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, A> Iterator for IterMut<'a, A> {
    type Item = &'a mut A;

    #[inline]
    fn next(&mut self) -> Option<&'a mut A> { self.inner.next() }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) { self.inner.size_hint() }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, A> DoubleEndedIterator for IterMut<'a, A> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a mut A> { self.inner.next_back() }
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
///
/// [`Option`]: enum.Option.html
/// [`Some`]: enum.Option.html#variant.Some
/// [`Option::into_iter`]: enum.Option.html#method.into_iter
#[derive(Clone, Debug)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct IntoIter<A> { inner: Item<A> }

#[stable(feature = "rust1", since = "1.0.0")]
impl<A> Iterator for IntoIter<A> {
    type Item = A;

    #[inline]
    fn next(&mut self) -> Option<A> { self.inner.next() }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) { self.inner.size_hint() }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A> DoubleEndedIterator for IntoIter<A> {
    #[inline]
    fn next_back(&mut self) -> Option<A> { self.inner.next_back() }
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
    ///
    /// [`Iterator`]: ../iter/trait.Iterator.html
    #[inline]
    fn from_iter<I: IntoIterator<Item=Option<A>>>(iter: I) -> Option<V> {
        // FIXME(#11084): This could be replaced with Iterator::scan when this
        // performance bug is closed.

        struct Adapter<Iter> {
            iter: Iter,
            found_none: bool,
        }

        impl<T, Iter: Iterator<Item=Option<T>>> Iterator for Adapter<Iter> {
            type Item = T;

            #[inline]
            fn next(&mut self) -> Option<T> {
                match self.iter.next() {
                    Some(Some(value)) => Some(value),
                    Some(None) => {
                        self.found_none = true;
                        None
                    }
                    None => None,
                }
            }

            #[inline]
            fn size_hint(&self) -> (usize, Option<usize>) {
                if self.found_none {
                    (0, Some(0))
                } else {
                    let (_, upper) = self.iter.size_hint();
                    (0, upper)
                }
            }
        }

        let mut adapter = Adapter { iter: iter.into_iter(), found_none: false };
        let v: V = FromIterator::from_iter(adapter.by_ref());

        if adapter.found_none {
            None
        } else {
            Some(v)
        }
    }
}

/// The error type that results from applying the try operator (`?`) to a `None` value. If you wish
/// to allow `x?` (where `x` is an `Option<T>`) to be converted into your error type, you can
/// implement `impl From<NoneError>` for `YourErrorType`. In that case, `x?` within a function that
/// returns `Result<_, YourErrorType>` will translate a `None` value into an `Err` result.
#[unstable(feature = "try_trait", issue = "42327")]
#[derive(Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Debug, Hash)]
pub struct NoneError;

#[unstable(feature = "try_trait", issue = "42327")]
impl<T> ops::Try for Option<T> {
    type Ok = T;
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

impl<T> Option<Option<T>> {
    /// Converts from `Option<Option<T>>` to `Option<T>`
    ///
    /// # Examples
    /// Basic usage:
    /// ```
    /// #![feature(option_flattening)]
    /// let x: Option<Option<u32>> = Some(Some(6));
    /// assert_eq!(Some(6), x.flatten());
    ///
    /// let x: Option<Option<u32>> = Some(None);
    /// assert_eq!(None, x.flatten());
    ///
    /// let x: Option<Option<u32>> = None;
    /// assert_eq!(None, x.flatten());
    /// ```
    /// Flattening once only removes one level of nesting:
    /// ```
    /// #![feature(option_flattening)]
    /// let x: Option<Option<Option<u32>>> = Some(Some(Some(6)));
    /// assert_eq!(Some(Some(6)), x.flatten());
    /// assert_eq!(Some(6), x.flatten().flatten());
    /// ```
    #[inline]
    #[unstable(feature = "option_flattening", issue = "60258")]
    pub fn flatten(self) -> Option<T> {
        self.and_then(convert::identity)
    }
}
