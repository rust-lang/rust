// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Traits for working with Errors.
//!
//! # The `Error` trait
//!
//! `Error` is a trait representing the basic expectations for error values,
//! i.e. values of type `E` in `Result<T, E>`. At a minimum, errors must provide
//! a description, but they may optionally provide additional detail (via
//! `Display`) and cause chain information:
//!
//! ```
//! use std::fmt::Display;
//!
//! trait Error: Display {
//!     fn description(&self) -> &str;
//!
//!     fn cause(&self) -> Option<&Error> { None }
//! }
//! ```
//!
//! The `cause` method is generally used when errors cross "abstraction
//! boundaries", i.e.  when a one module must report an error that is "caused"
//! by an error from a lower-level module. This setup makes it possible for the
//! high-level module to provide its own errors that do not commit to any
//! particular implementation, but also reveal some of its implementation for
//! debugging via `cause` chains.

#![stable(feature = "rust1", since = "1.0.0")]

// A note about crates and the facade:
//
// Originally, the `Error` trait was defined in libcore, and the impls
// were scattered about. However, coherence objected to this
// arrangement, because to create the blanket impls for `Box` required
// knowing that `&str: !Error`, and we have no means to deal with that
// sort of conflict just now. Therefore, for the time being, we have
// moved the `Error` trait into libstd. As we evolve a sol'n to the
// coherence challenge (e.g., specialization, neg impls, etc) we can
// reconsider what crate these items belong in.

use boxed::Box;
use convert::From;
use fmt::{self, Debug, Display};
use marker::Send;
use num;
use option::Option;
use option::Option::None;
use str;
use string::{self, String};

/// Base functionality for all errors in Rust.
#[stable(feature = "rust1", since = "1.0.0")]
pub trait Error: Debug + Display {
    /// A short description of the error.
    ///
    /// The description should not contain newlines or sentence-ending
    /// punctuation, to facilitate embedding in larger user-facing
    /// strings.
    #[stable(feature = "rust1", since = "1.0.0")]
    fn description(&self) -> &str;

    /// The lower-level cause of this error, if any.
    #[stable(feature = "rust1", since = "1.0.0")]
    fn cause(&self) -> Option<&Error> { None }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, E: Error + 'a> From<E> for Box<Error + 'a> {
    fn from(err: E) -> Box<Error + 'a> {
        Box::new(err)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, E: Error + Send + 'a> From<E> for Box<Error + Send + 'a> {
    fn from(err: E) -> Box<Error + Send + 'a> {
        Box::new(err)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl From<String> for Box<Error + Send> {
    fn from(err: String) -> Box<Error + Send> {
        #[derive(Debug)]
        struct StringError(String);

        impl Error for StringError {
            fn description(&self) -> &str { &self.0 }
        }

        impl Display for StringError {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                Display::fmt(&self.0, f)
            }
        }

        Box::new(StringError(err))
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, 'b> From<&'b str> for Box<Error + Send + 'a> {
    fn from(err: &'b str) -> Box<Error + Send + 'a> {
        From::from(String::from_str(err))
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Error for str::ParseBoolError {
    fn description(&self) -> &str { "failed to parse bool" }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Error for str::Utf8Error {
    fn description(&self) -> &str {
        "invalid utf-8: corrupt contents"
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Error for num::ParseIntError {
    fn description(&self) -> &str {
        self.description()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Error for num::ParseFloatError {
    fn description(&self) -> &str {
        self.description()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Error for string::FromUtf8Error {
    fn description(&self) -> &str {
        "invalid utf-8"
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Error for string::FromUtf16Error {
    fn description(&self) -> &str {
        "invalid utf-16"
    }
}

