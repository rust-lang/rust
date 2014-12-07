// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A trait for converting between different types of errors.
//!
//! # The `FromError` trait
//!
//! `FromError` is a simple trait that expresses conversions between different
//! error types. To provide maximum flexibility, it does not require either of
//! the types to actually implement the `Error` trait from the `std` crate,
//! although this will be the common case.
//!
//! The main use of this trait is in the `try!` macro from the `std` crate,
//! which uses it to automatically convert a given error to the error specified
//! in a function's return type.

/// A trait for types that can be converted from a given error type `E`.
pub trait FromError<E> {
    /// Perform the conversion.
    fn from_error(err: E) -> Self;
}

// Any type is convertable from itself
impl<E> FromError<E> for E {
    fn from_error(err: E) -> E {
        err
    }
}
