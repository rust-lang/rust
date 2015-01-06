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
//! a description, but they may optionally provide additional detail and cause
//! chain information:
//!
//! ```
//! trait Error: Send {
//!     fn description(&self) -> &str;
//!
//!     fn detail(&self) -> Option<String> { None }
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
//!
//! # The `FromError` trait
//!
//! `FromError` is a simple trait that expresses conversions between different
//! error types. To provide maximum flexibility, it does not require either of
//! the types to actually implement the `Error` trait, although this will be the
//! common case.
//!
//! The main use of this trait is in the `try!` macro, which uses it to
//! automatically convert a given error to the error specified in a function's
//! return type.
//!
//! For example,
//!
//! ```
//! use std::error::FromError;
//! use std::io::{File, IoError};
//! use std::os::{MemoryMap, MapError};
//! use std::path::Path;
//!
//! enum MyError {
//!     Io(IoError),
//!     Map(MapError)
//! }
//!
//! impl FromError<IoError> for MyError {
//!     fn from_error(err: IoError) -> MyError {
//!         MyError::Io(err)
//!     }
//! }
//!
//! impl FromError<MapError> for MyError {
//!     fn from_error(err: MapError) -> MyError {
//!         MyError::Map(err)
//!     }
//! }
//!
//! #[allow(unused_variables)]
//! fn open_and_map() -> Result<(), MyError> {
//!     let f = try!(File::open(&Path::new("foo.txt")));
//!     let m = try!(MemoryMap::new(0, &[]));
//!     // do something interesting here...
//!     Ok(())
//! }
//! ```

#![stable]

use prelude::v1::*;

use str::Utf8Error;
use string::{FromUtf8Error, FromUtf16Error};

/// Base functionality for all errors in Rust.
#[unstable = "the exact API of this trait may change"]
pub trait Error: Send {
    /// A short description of the error; usually a static string.
    fn description(&self) -> &str;

    /// A detailed description of the error, usually including dynamic information.
    fn detail(&self) -> Option<String> { None }

    /// The lower-level cause of this error, if any.
    fn cause(&self) -> Option<&Error> { None }
}

/// A trait for types that can be converted from a given error type `E`.
#[stable]
pub trait FromError<E> {
    /// Perform the conversion.
    fn from_error(err: E) -> Self;
}

// Any type is convertable from itself
#[stable]
impl<E> FromError<E> for E {
    fn from_error(err: E) -> E {
        err
    }
}

#[stable]
impl Error for Utf8Error {
    fn description(&self) -> &str {
        match *self {
            Utf8Error::TooShort => "invalid utf-8: not enough bytes",
            Utf8Error::InvalidByte(..) => "invalid utf-8: corrupt contents",
        }
    }

    fn detail(&self) -> Option<String> { Some(self.to_string()) }
}

#[stable]
impl Error for FromUtf8Error {
    fn description(&self) -> &str { "invalid utf-8" }
    fn detail(&self) -> Option<String> { Some(self.to_string()) }
}

#[stable]
impl Error for FromUtf16Error {
    fn description(&self) -> &str { "invalid utf-16" }
}
