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

use any::TypeId;
use boxed::{self, Box};
use convert::From;
use fmt::{self, Debug, Display};
use marker::{Send, Sync, Reflect};
use mem::transmute;
use num;
use option::Option::{self, Some, None};
use result::Result::{self, Ok, Err};
use raw::TraitObject;
use str;
use string::{self, String};

/// Base functionality for all errors in Rust.
#[stable(feature = "rust1", since = "1.0.0")]
pub trait Error: Debug + Display + Reflect {
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

    /// Get the `TypeId` of `self`
    #[doc(hidden)]
    #[unstable(feature = "core",
               reason = "unclear whether to commit to this public implementation detail")]
    fn type_id(&self) -> TypeId where Self: 'static {
        TypeId::of::<Self>()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, E: Error + 'a> From<E> for Box<Error + 'a> {
    fn from(err: E) -> Box<Error + 'a> {
        Box::new(err)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, E: Error + Send + Sync + 'a> From<E> for Box<Error + Send + Sync + 'a> {
    fn from(err: E) -> Box<Error + Send + Sync + 'a> {
        Box::new(err)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl From<String> for Box<Error + Send + Sync> {
    fn from(err: String) -> Box<Error + Send + Sync> {
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
impl<'a, 'b> From<&'b str> for Box<Error + Send + Sync + 'a> {
    fn from(err: &'b str) -> Box<Error + Send + Sync + 'a> {
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
        self.__description()
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

// copied from any.rs
impl Error + 'static {
    /// Returns true if the boxed type is the same as `T`
    #[unstable(feature = "error_downcast", reason = "recently added")]
    #[inline]
    pub fn is<T: Error + 'static>(&self) -> bool {
        // Get TypeId of the type this function is instantiated with
        let t = TypeId::of::<T>();

        // Get TypeId of the type in the trait object
        let boxed = self.type_id();

        // Compare both TypeIds on equality
        t == boxed
    }

    /// Returns some reference to the boxed value if it is of type `T`, or
    /// `None` if it isn't.
    #[unstable(feature = "error_downcast", reason = "recently added")]
    #[inline]
    pub fn downcast_ref<T: Error + 'static>(&self) -> Option<&T> {
        if self.is::<T>() {
            unsafe {
                // Get the raw representation of the trait object
                let to: TraitObject = transmute(self);

                // Extract the data pointer
                Some(transmute(to.data))
            }
        } else {
            None
        }
    }

    /// Returns some mutable reference to the boxed value if it is of type `T`, or
    /// `None` if it isn't.
    #[unstable(feature = "error_downcast", reason = "recently added")]
    #[inline]
    pub fn downcast_mut<T: Error + 'static>(&mut self) -> Option<&mut T> {
        if self.is::<T>() {
            unsafe {
                // Get the raw representation of the trait object
                let to: TraitObject = transmute(self);

                // Extract the data pointer
                Some(transmute(to.data))
            }
        } else {
            None
        }
    }
}

impl Error + 'static + Send {
    /// Forwards to the method defined on the type `Any`.
    #[unstable(feature = "error_downcast", reason = "recently added")]
    #[inline]
    pub fn is<T: Error + 'static>(&self) -> bool {
        <Error + 'static>::is::<T>(self)
    }

    /// Forwards to the method defined on the type `Any`.
    #[unstable(feature = "error_downcast", reason = "recently added")]
    #[inline]
    pub fn downcast_ref<T: Error + 'static>(&self) -> Option<&T> {
        <Error + 'static>::downcast_ref::<T>(self)
    }

    /// Forwards to the method defined on the type `Any`.
    #[unstable(feature = "error_downcast", reason = "recently added")]
    #[inline]
    pub fn downcast_mut<T: Error + 'static>(&mut self) -> Option<&mut T> {
        <Error + 'static>::downcast_mut::<T>(self)
    }
}

impl Error {
    #[inline]
    #[unstable(feature = "error_downcast", reason = "recently added")]
    /// Attempt to downcast the box to a concrete type.
    pub fn downcast<T: Error + 'static>(self: Box<Self>) -> Result<Box<T>, Box<Error>> {
        if self.is::<T>() {
            unsafe {
                // Get the raw representation of the trait object
                let raw = boxed::into_raw(self);
                let to: TraitObject =
                    transmute::<*mut Error, TraitObject>(raw);

                // Extract the data pointer
                Ok(Box::from_raw(to.data as *mut T))
            }
        } else {
            Err(self)
        }
    }
}

impl Error + Send {
    #[inline]
    #[unstable(feature = "error_downcast", reason = "recently added")]
    /// Attempt to downcast the box to a concrete type.
    pub fn downcast<T: Error + 'static>(self: Box<Self>) -> Result<Box<T>, Box<Error + Send>> {
        let err: Box<Error> = self;
        <Error>::downcast(err).map_err(|s| unsafe {
            // reapply the Send marker
            transmute::<Box<Error>, Box<Error + Send>>(s)
        })
    }
}
