// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Optional values
//!
//! Type `Option` represents an optional value: every `Option`
//! is either `Some` and contains a value, or `None`, and
//! does not. `Option` types are very common in Rust code, as
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
//! Options are commonly paired with pattern matching to query the presence
//! of a value and take action, always accounting for the `None` case.
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
//!     None    => println!("Cannot divide by 0")
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
//! the optional owned box, `Option<Box<T>>`.
//!
//! The following example uses `Option` to create an optional box of
//! `int`. Notice that in order to use the inner `int` value first the
//! `check_optional` function needs to use pattern matching to
//! determine whether the box has a value (i.e. it is `Some(...)`) or
//! not (`None`).
//!
//! ```
//! let optional: Option<Box<int>> = None;
//! check_optional(&optional);
//!
//! let optional: Option<Box<int>> = Some(box 9000);
//! check_optional(&optional);
//!
//! fn check_optional(optional: &Option<Box<int>>) {
//!     match *optional {
//!         Some(ref p) => println!("have value {}", p),
//!         None => println!("have no value")
//!     }
//! }
//! ```
//!
//! This usage of `Option` to create safe nullable pointers is so
//! common that Rust does special optimizations to make the
//! representation of `Option<Box<T>>` a single pointer. Optional pointers
//! in Rust are stored as efficiently as any other pointer type.
//!
//! # Examples
//!
//! Basic pattern matching on `Option`:
//!
//! ```
//! let msg = Some("howdy");
//!
//! // Take a reference to the contained string
//! match msg {
//!     Some(ref m) => println!("{}", *m),
//!     None => ()
//! }
//!
//! // Remove the contained string, destroying the Option
//! let unwrapped_msg = match msg {
//!     Some(m) => m,
//!     None => "default message"
//! };
//! ```
//!
//! Initialize a result to `None` before a loop:
//!
//! ```
//! enum Kingdom { Plant(uint, &'static str), Animal(uint, &'static str) }
//!
//! // A list of data to search through.
//! let all_the_big_things = [
//!     Plant(250, "redwood"),
//!     Plant(230, "noble fir"),
//!     Plant(229, "sugar pine"),
//!     Animal(25, "blue whale"),
//!     Animal(19, "fin whale"),
//!     Animal(15, "north pacific right whale"),
//! ];
//!
//! // We're going to search for the name of the biggest animal,
//! // but to start with we've just got `None`.
//! let mut name_of_biggest_animal = None;
//! let mut size_of_biggest_animal = 0;
//! for big_thing in all_the_big_things.iter() {
//!     match *big_thing {
//!         Animal(size, name) if size > size_of_biggest_animal => {
//!             // Now we've found the name of some big animal
//!             size_of_biggest_animal = size;
//!             name_of_biggest_animal = Some(name);
//!         }
//!         Animal(..) | Plant(..) => ()
//!     }
//! }
//!
//! match name_of_biggest_animal {
//!     Some(name) => println!("the biggest animal is {}", name),
//!     None => println!("there are no animals :(")
//! }
//! ```

use any::Any;
use kinds::Send;

pub use core::option::{Option, Some, None, Item, collect};

/// Extension trait for the `Option` type to add an `expect` method

// FIXME(#14008) should this trait even exist?
pub trait Expect<T> {
    /// Unwraps an option, yielding the content of a `Some`
    ///
    /// # Failure
    ///
    /// Fails if the value is a `None` with a custom failure message provided by
    /// `msg`.
    fn expect<M: Any + Send>(self, m: M) -> T;
}

impl<T> Expect<T> for Option<T> {
    #[inline]
    fn expect<M: Any + Send>(self, msg: M) -> T {
        match self {
            Some(val) => val,
            None => fail!(msg),
        }
    }
}
