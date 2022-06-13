//! Interfaces for working with Errors.
//!
//! # Error Handling In Rust
//!
//! The Rust language provides two complementary systems for constructing /
//! representing, reporting, propagating, reacting to, and discarding errors.
//! These responsibilities are collectively known as "error handling." The
//! components of the first system, the panic runtime and interfaces, are most
//! commonly used to represent bugs that have been detected in your program. The
//! components of the second system, `Result`, the error traits, and user
//! defined types, are used to represent anticipated runtime failure modes of
//! your program.
//!
//! ## The Panic Interfaces
//!
//! The following are the primary interfaces of the panic system and the
//! responsibilities they cover:
//!
//! * [`panic!`] and [`panic_any`] (Constructing, Propagated automatically)
//! * [`PanicInfo`] (Reporting)
//! * [`set_hook`], [`take_hook`], and [`#[panic_handler]`][panic-handler] (Reporting)
//! * [`catch_unwind`] and [`resume_unwind`] (Discarding, Propagating)
//!
//! The following are the primary interfaces of the error system and the
//! responsibilities they cover:
//!
//! * [`Result`] (Propagating, Reacting)
//! * The [`Error`] trait (Reporting)
//! * User defined types (Constructing / Representing)
//! * [`match`] and [`downcast`] (Reacting)
//! * The question mark operator ([`?`]) (Propagating)
//! * The partially stable [`Try`] traits (Propagating, Constructing)
//! * [`Termination`] (Reporting)
//!
//! ## Converting Errors into Panics
//!
//! The panic and error systems are not entirely distinct. Often times errors
//! that are anticipated runtime failures in an API might instead represent bugs
//! to a caller. For these situations the standard library provides APIs for
//! constructing panics with an `Error` as it's source.
//!
//! * [`Result::unwrap`]
//! * [`Result::expect`]
//!
//! These functions are equivalent, they either return the inner value if the
//! `Result` is `Ok` or panic if the `Result` is `Err` printing the inner error
//! as the source. The only difference between them is that with `expect` you
//! provide a panic error message to be printed alongside the source, whereas
//! `unwrap` has a default message indicating only that you unwraped an `Err`.
//!
//! Of the two, `expect` is generally preferred since its `msg` field allows you
//! to convey your intent and assumptions which makes tracking down the source
//! of a panic easier. `unwrap` on the other hand can still be a good fit in
//! situations where you can trivially show that a piece of code will never
//! panic, such as `"127.0.0.1".parse::<std::net::IpAddr>().unwrap()` or early
//! prototyping.
//!
//! # Common Message Styles
//!
//! There are two common styles for how people word `expect` messages. Using
//! the message to present information to users encountering a panic
//! ("expect as error message") or using the message to present information
//! to developers debugging the panic ("expect as precondition").
//!
//! In the former case the expect message is used to describe the error that
//! has occurred which is considered a bug. Consider the following example:
//!
//! ```should_panic
//! // Read environment variable, panic if it is not present
//! let path = std::env::var("IMPORTANT_PATH").unwrap();
//! ```
//!
//! In the "expect as error message" style we would use expect to describe
//! that the environment variable was not set when it should have been:
//!
//! ```should_panic
//! let path = std::env::var("IMPORTANT_PATH")
//!     .expect("env variable `IMPORTANT_PATH` is not set");
//! ```
//!
//! In the "expect as precondition" style, we would instead describe the
//! reason we _expect_ the `Result` should be `Ok`. With this style we would
//! prefer to write:
//!
//! ```should_panic
//! let path = std::env::var("IMPORTANT_PATH")
//!     .expect("env variable `IMPORTANT_PATH` should be set by `wrapper_script.sh`");
//! ```
//!
//! The "expect as error message" style does not work as well with the
//! default output of the std panic hooks, and often ends up repeating
//! information that is already communicated by the source error being
//! unwrapped:
//!
//! ```text
//! thread 'main' panicked at 'env variable `IMPORTANT_PATH` is not set: NotPresent', src/main.rs:4:6
//! ```
//!
//! In this example we end up mentioning that an env variable is not set,
//! followed by our source message that says the env is not present, the
//! only additional information we're communicating is the name of the
//! environment variable being checked.
//!
//! The "expect as precondition" style instead focuses on source code
//! readability, making it easier to understand what must have gone wrong in
//! situations where panics are being used to represent bugs exclusively.
//! Also, by framing our expect in terms of what "SHOULD" have happened to
//! prevent the source error, we end up introducing new information that is
//! independent from our source error.
//!
//! ```text
//! thread 'main' panicked at 'env variable `IMPORTANT_PATH` should be set by `wrapper_script.sh`: NotPresent', src/main.rs:4:6
//! ```
//!
//! In this example we are communicating not only the name of the
//! environment variable that should have been set, but also an explanation
//! for why it should have been set, and we let the source error display as
//! a clear contradiction to our expectation.
//!
//! **Hint**: If you're having trouble remembering how to phrase
//! expect-as-precondition style error messages remember to focus on the word
//! "should" as in "env variable should be set by blah" or "the given binary
//! should be available and executable by the current user".
//!
//! [`panic_any`]: crate::panic::panic_any
//! [`PanicInfo`]: crate::panic::PanicInfo
//! [`catch_unwind`]: crate::panic::catch_unwind
//! [`resume_unwind`]: crate::panic::resume_unwind
//! [`downcast`]: crate::error::Error
//! [`Termination`]: crate::process::Termination
//! [`Try`]: crate::ops::Try
//! [panic hook]: crate::panic::set_hook
//! [`set_hook`]: crate::panic::set_hook
//! [`take_hook`]: crate::panic::take_hook
//! [panic-handler]: <https://doc.rust-lang.org/nomicon/panic-handler.html>
//! [`match`]: ../../std/keyword.match.html
//! [`?`]: ../../std/result/index.html#the-question-mark-operator-

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

#[cfg(test)]
mod tests;

use core::array;
use core::convert::Infallible;

use crate::alloc::{AllocError, LayoutError};
use crate::any::{Demand, Provider, TypeId};
use crate::backtrace::Backtrace;
use crate::borrow::Cow;
use crate::cell;
use crate::char;
use crate::fmt::{self, Debug, Display, Write};
use crate::io;
use crate::mem::transmute;
use crate::num;
use crate::str;
use crate::string;
use crate::sync::Arc;
use crate::time;

/// `Error` is a trait representing the basic expectations for error values,
/// i.e., values of type `E` in [`Result<T, E>`].
///
/// Errors must describe themselves through the [`Display`] and [`Debug`]
/// traits. Error messages are typically concise lowercase sentences without
/// trailing punctuation:
///
/// ```
/// let err = "NaN".parse::<u32>().unwrap_err();
/// assert_eq!(err.to_string(), "invalid digit found in string");
/// ```
///
/// Errors may provide cause chain information. [`Error::source()`] is generally
/// used when errors cross "abstraction boundaries". If one module must report
/// an error that is caused by an error from a lower-level module, it can allow
/// accessing that error via [`Error::source()`]. This makes it possible for the
/// high-level module to provide its own errors while also revealing some of the
/// implementation for debugging via `source` chains.
#[stable(feature = "rust1", since = "1.0.0")]
#[cfg_attr(not(test), rustc_diagnostic_item = "Error")]
pub trait Error: Debug + Display {
    /// The lower-level source of this error, if any.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::error::Error;
    /// use std::fmt;
    ///
    /// #[derive(Debug)]
    /// struct SuperError {
    ///     source: SuperErrorSideKick,
    /// }
    ///
    /// impl fmt::Display for SuperError {
    ///     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    ///         write!(f, "SuperError is here!")
    ///     }
    /// }
    ///
    /// impl Error for SuperError {
    ///     fn source(&self) -> Option<&(dyn Error + 'static)> {
    ///         Some(&self.source)
    ///     }
    /// }
    ///
    /// #[derive(Debug)]
    /// struct SuperErrorSideKick;
    ///
    /// impl fmt::Display for SuperErrorSideKick {
    ///     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    ///         write!(f, "SuperErrorSideKick is here!")
    ///     }
    /// }
    ///
    /// impl Error for SuperErrorSideKick {}
    ///
    /// fn get_super_error() -> Result<(), SuperError> {
    ///     Err(SuperError { source: SuperErrorSideKick })
    /// }
    ///
    /// fn main() {
    ///     match get_super_error() {
    ///         Err(e) => {
    ///             println!("Error: {e}");
    ///             println!("Caused by: {}", e.source().unwrap());
    ///         }
    ///         _ => println!("No error"),
    ///     }
    /// }
    /// ```
    #[stable(feature = "error_source", since = "1.30.0")]
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        None
    }

    /// Gets the `TypeId` of `self`.
    #[doc(hidden)]
    #[unstable(
        feature = "error_type_id",
        reason = "this is memory-unsafe to override in user code",
        issue = "60784"
    )]
    fn type_id(&self, _: private::Internal) -> TypeId
    where
        Self: 'static,
    {
        TypeId::of::<Self>()
    }

    /// Returns a stack backtrace, if available, of where this error occurred.
    ///
    /// This function allows inspecting the location, in code, of where an error
    /// happened. The returned `Backtrace` contains information about the stack
    /// trace of the OS thread of execution of where the error originated from.
    ///
    /// Note that not all errors contain a `Backtrace`. Also note that a
    /// `Backtrace` may actually be empty. For more information consult the
    /// `Backtrace` type itself.
    #[unstable(feature = "backtrace", issue = "53487")]
    fn backtrace(&self) -> Option<&Backtrace> {
        None
    }

    /// ```
    /// if let Err(e) = "xc".parse::<u32>() {
    ///     // Print `e` itself, no need for description().
    ///     eprintln!("Error: {e}");
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[deprecated(since = "1.42.0", note = "use the Display impl or to_string()")]
    fn description(&self) -> &str {
        "description() is deprecated; use Display"
    }

    #[stable(feature = "rust1", since = "1.0.0")]
    #[deprecated(
        since = "1.33.0",
        note = "replaced by Error::source, which can support downcasting"
    )]
    #[allow(missing_docs)]
    fn cause(&self) -> Option<&dyn Error> {
        self.source()
    }

    /// Provides type based access to context intended for error reports.
    ///
    /// Used in conjunction with [`context`] and [`context_ref`] to extract
    /// references to member variables from `dyn Error` trait objects.
    ///
    /// # Example
    ///
    /// ```rust
    /// #![feature(provide_any)]
    /// #![feature(error_in_core)]
    /// use core::fmt;
    /// use core::any::Demand;
    ///
    /// #[derive(Debug)]
    /// struct MyBacktrace {
    ///     // ...
    /// }
    ///
    /// impl MyBacktrace {
    ///     fn new() -> MyBacktrace {
    ///         // ...
    ///         # MyBacktrace {}
    ///     }
    /// }
    ///
    /// #[derive(Debug)]
    /// struct SourceError {
    ///     // ...
    /// }
    ///
    /// impl fmt::Display for SourceError {
    ///     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    ///         write!(f, "Example Source Error")
    ///     }
    /// }
    ///
    /// impl std::error::Error for SourceError {}
    ///
    /// #[derive(Debug)]
    /// struct Error {
    ///     source: SourceError,
    ///     backtrace: MyBacktrace,
    /// }
    ///
    /// impl fmt::Display for Error {
    ///     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    ///         write!(f, "Example Error")
    ///     }
    /// }
    ///
    /// impl std::error::Error for Error {
    ///     fn provide<'a>(&'a self, req: &mut Demand<'a>) {
    ///         req
    ///             .provide_ref::<MyBacktrace>(&self.backtrace)
    ///             .provide_ref::<dyn std::error::Error + 'static>(&self.source);
    ///     }
    /// }
    ///
    /// fn main() {
    ///     let backtrace = MyBacktrace::new();
    ///     let source = SourceError {};
    ///     let error = Error { source, backtrace };
    ///     let dyn_error = &error as &dyn std::error::Error;
    ///     let backtrace_ref = dyn_error.request_ref::<MyBacktrace>().unwrap();
    ///
    ///     assert!(core::ptr::eq(&error.backtrace, backtrace_ref));
    /// }
    /// ```
    #[unstable(feature = "generic_member_access", issue = "none")]
    fn provide<'a>(&'a self, _req: &mut Demand<'a>) {}
}

#[unstable(feature = "generic_member_access", issue = "none")]
impl Provider for dyn Error + 'static {
    fn provide<'a>(&'a self, req: &mut Demand<'a>) {
        self.provide(req)
    }
}

mod private {
    // This is a hack to prevent `type_id` from being overridden by `Error`
    // implementations, since that can enable unsound downcasting.
    #[unstable(feature = "error_type_id", issue = "60784")]
    #[derive(Debug)]
    pub struct Internal;
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, E: Error + 'a> From<E> for Box<dyn Error + 'a> {
    /// Converts a type of [`Error`] into a box of dyn [`Error`].
    ///
    /// # Examples
    ///
    /// ```
    /// use std::error::Error;
    /// use std::fmt;
    /// use std::mem;
    ///
    /// #[derive(Debug)]
    /// struct AnError;
    ///
    /// impl fmt::Display for AnError {
    ///     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    ///         write!(f, "An error")
    ///     }
    /// }
    ///
    /// impl Error for AnError {}
    ///
    /// let an_error = AnError;
    /// assert!(0 == mem::size_of_val(&an_error));
    /// let a_boxed_error = Box::<dyn Error>::from(an_error);
    /// assert!(mem::size_of::<Box<dyn Error>>() == mem::size_of_val(&a_boxed_error))
    /// ```
    fn from(err: E) -> Box<dyn Error + 'a> {
        Box::new(err)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, E: Error + Send + Sync + 'a> From<E> for Box<dyn Error + Send + Sync + 'a> {
    /// Converts a type of [`Error`] + [`Send`] + [`Sync`] into a box of
    /// dyn [`Error`] + [`Send`] + [`Sync`].
    ///
    /// # Examples
    ///
    /// ```
    /// use std::error::Error;
    /// use std::fmt;
    /// use std::mem;
    ///
    /// #[derive(Debug)]
    /// struct AnError;
    ///
    /// impl fmt::Display for AnError {
    ///     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    ///         write!(f, "An error")
    ///     }
    /// }
    ///
    /// impl Error for AnError {}
    ///
    /// unsafe impl Send for AnError {}
    ///
    /// unsafe impl Sync for AnError {}
    ///
    /// let an_error = AnError;
    /// assert!(0 == mem::size_of_val(&an_error));
    /// let a_boxed_error = Box::<dyn Error + Send + Sync>::from(an_error);
    /// assert!(
    ///     mem::size_of::<Box<dyn Error + Send + Sync>>() == mem::size_of_val(&a_boxed_error))
    /// ```
    fn from(err: E) -> Box<dyn Error + Send + Sync + 'a> {
        Box::new(err)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl From<String> for Box<dyn Error + Send + Sync> {
    /// Converts a [`String`] into a box of dyn [`Error`] + [`Send`] + [`Sync`].
    ///
    /// # Examples
    ///
    /// ```
    /// use std::error::Error;
    /// use std::mem;
    ///
    /// let a_string_error = "a string error".to_string();
    /// let a_boxed_error = Box::<dyn Error + Send + Sync>::from(a_string_error);
    /// assert!(
    ///     mem::size_of::<Box<dyn Error + Send + Sync>>() == mem::size_of_val(&a_boxed_error))
    /// ```
    #[inline]
    fn from(err: String) -> Box<dyn Error + Send + Sync> {
        struct StringError(String);

        impl Error for StringError {
            #[allow(deprecated)]
            fn description(&self) -> &str {
                &self.0
            }
        }

        impl Display for StringError {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                Display::fmt(&self.0, f)
            }
        }

        // Purposefully skip printing "StringError(..)"
        impl Debug for StringError {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                Debug::fmt(&self.0, f)
            }
        }

        Box::new(StringError(err))
    }
}

#[stable(feature = "string_box_error", since = "1.6.0")]
impl From<String> for Box<dyn Error> {
    /// Converts a [`String`] into a box of dyn [`Error`].
    ///
    /// # Examples
    ///
    /// ```
    /// use std::error::Error;
    /// use std::mem;
    ///
    /// let a_string_error = "a string error".to_string();
    /// let a_boxed_error = Box::<dyn Error>::from(a_string_error);
    /// assert!(mem::size_of::<Box<dyn Error>>() == mem::size_of_val(&a_boxed_error))
    /// ```
    fn from(str_err: String) -> Box<dyn Error> {
        let err1: Box<dyn Error + Send + Sync> = From::from(str_err);
        let err2: Box<dyn Error> = err1;
        err2
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> From<&str> for Box<dyn Error + Send + Sync + 'a> {
    /// Converts a [`str`] into a box of dyn [`Error`] + [`Send`] + [`Sync`].
    ///
    /// [`str`]: prim@str
    ///
    /// # Examples
    ///
    /// ```
    /// use std::error::Error;
    /// use std::mem;
    ///
    /// let a_str_error = "a str error";
    /// let a_boxed_error = Box::<dyn Error + Send + Sync>::from(a_str_error);
    /// assert!(
    ///     mem::size_of::<Box<dyn Error + Send + Sync>>() == mem::size_of_val(&a_boxed_error))
    /// ```
    #[inline]
    fn from(err: &str) -> Box<dyn Error + Send + Sync + 'a> {
        From::from(String::from(err))
    }
}

#[stable(feature = "string_box_error", since = "1.6.0")]
impl From<&str> for Box<dyn Error> {
    /// Converts a [`str`] into a box of dyn [`Error`].
    ///
    /// [`str`]: prim@str
    ///
    /// # Examples
    ///
    /// ```
    /// use std::error::Error;
    /// use std::mem;
    ///
    /// let a_str_error = "a str error";
    /// let a_boxed_error = Box::<dyn Error>::from(a_str_error);
    /// assert!(mem::size_of::<Box<dyn Error>>() == mem::size_of_val(&a_boxed_error))
    /// ```
    fn from(err: &str) -> Box<dyn Error> {
        From::from(String::from(err))
    }
}

#[stable(feature = "cow_box_error", since = "1.22.0")]
impl<'a, 'b> From<Cow<'b, str>> for Box<dyn Error + Send + Sync + 'a> {
    /// Converts a [`Cow`] into a box of dyn [`Error`] + [`Send`] + [`Sync`].
    ///
    /// # Examples
    ///
    /// ```
    /// use std::error::Error;
    /// use std::mem;
    /// use std::borrow::Cow;
    ///
    /// let a_cow_str_error = Cow::from("a str error");
    /// let a_boxed_error = Box::<dyn Error + Send + Sync>::from(a_cow_str_error);
    /// assert!(
    ///     mem::size_of::<Box<dyn Error + Send + Sync>>() == mem::size_of_val(&a_boxed_error))
    /// ```
    fn from(err: Cow<'b, str>) -> Box<dyn Error + Send + Sync + 'a> {
        From::from(String::from(err))
    }
}

#[stable(feature = "cow_box_error", since = "1.22.0")]
impl<'a> From<Cow<'a, str>> for Box<dyn Error> {
    /// Converts a [`Cow`] into a box of dyn [`Error`].
    ///
    /// # Examples
    ///
    /// ```
    /// use std::error::Error;
    /// use std::mem;
    /// use std::borrow::Cow;
    ///
    /// let a_cow_str_error = Cow::from("a str error");
    /// let a_boxed_error = Box::<dyn Error>::from(a_cow_str_error);
    /// assert!(mem::size_of::<Box<dyn Error>>() == mem::size_of_val(&a_boxed_error))
    /// ```
    fn from(err: Cow<'a, str>) -> Box<dyn Error> {
        From::from(String::from(err))
    }
}

#[unstable(feature = "never_type", issue = "35121")]
impl Error for ! {}

#[unstable(
    feature = "allocator_api",
    reason = "the precise API and guarantees it provides may be tweaked.",
    issue = "32838"
)]
impl Error for AllocError {}

#[stable(feature = "alloc_layout", since = "1.28.0")]
impl Error for LayoutError {}

#[stable(feature = "rust1", since = "1.0.0")]
impl Error for str::ParseBoolError {
    #[allow(deprecated)]
    fn description(&self) -> &str {
        "failed to parse bool"
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Error for str::Utf8Error {
    #[allow(deprecated)]
    fn description(&self) -> &str {
        "invalid utf-8: corrupt contents"
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Error for num::ParseIntError {
    #[allow(deprecated)]
    fn description(&self) -> &str {
        self.__description()
    }
}

#[stable(feature = "try_from", since = "1.34.0")]
impl Error for num::TryFromIntError {
    #[allow(deprecated)]
    fn description(&self) -> &str {
        self.__description()
    }
}

#[stable(feature = "try_from", since = "1.34.0")]
impl Error for array::TryFromSliceError {
    #[allow(deprecated)]
    fn description(&self) -> &str {
        self.__description()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Error for num::ParseFloatError {
    #[allow(deprecated)]
    fn description(&self) -> &str {
        self.__description()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Error for string::FromUtf8Error {
    #[allow(deprecated)]
    fn description(&self) -> &str {
        "invalid utf-8"
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl Error for string::FromUtf16Error {
    #[allow(deprecated)]
    fn description(&self) -> &str {
        "invalid utf-16"
    }
}

#[stable(feature = "str_parse_error2", since = "1.8.0")]
impl Error for Infallible {
    fn description(&self) -> &str {
        match *self {}
    }
}

#[stable(feature = "decode_utf16", since = "1.9.0")]
impl Error for char::DecodeUtf16Error {
    #[allow(deprecated)]
    fn description(&self) -> &str {
        "unpaired surrogate found"
    }
}

#[stable(feature = "u8_from_char", since = "1.59.0")]
impl Error for char::TryFromCharError {}

#[unstable(feature = "map_try_insert", issue = "82766")]
impl<'a, K: Debug + Ord, V: Debug> Error
    for crate::collections::btree_map::OccupiedError<'a, K, V>
{
    #[allow(deprecated)]
    fn description(&self) -> &str {
        "key already exists"
    }
}

#[unstable(feature = "map_try_insert", issue = "82766")]
impl<'a, K: Debug, V: Debug> Error for crate::collections::hash_map::OccupiedError<'a, K, V> {
    #[allow(deprecated)]
    fn description(&self) -> &str {
        "key already exists"
    }
}

#[stable(feature = "box_error", since = "1.8.0")]
impl<T: Error> Error for Box<T> {
    #[allow(deprecated, deprecated_in_future)]
    fn description(&self) -> &str {
        Error::description(&**self)
    }

    #[allow(deprecated)]
    fn cause(&self) -> Option<&dyn Error> {
        Error::cause(&**self)
    }

    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Error::source(&**self)
    }
}

#[unstable(feature = "thin_box", issue = "92791")]
impl<T: ?Sized + crate::error::Error> crate::error::Error for crate::boxed::ThinBox<T> {
    fn source(&self) -> Option<&(dyn crate::error::Error + 'static)> {
        use core::ops::Deref;
        self.deref().source()
    }
}

#[stable(feature = "error_by_ref", since = "1.51.0")]
impl<'a, T: Error + ?Sized> Error for &'a T {
    #[allow(deprecated, deprecated_in_future)]
    fn description(&self) -> &str {
        Error::description(&**self)
    }

    #[allow(deprecated)]
    fn cause(&self) -> Option<&dyn Error> {
        Error::cause(&**self)
    }

    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Error::source(&**self)
    }

    fn backtrace(&self) -> Option<&Backtrace> {
        Error::backtrace(&**self)
    }
}

#[stable(feature = "arc_error", since = "1.52.0")]
impl<T: Error + ?Sized> Error for Arc<T> {
    #[allow(deprecated, deprecated_in_future)]
    fn description(&self) -> &str {
        Error::description(&**self)
    }

    #[allow(deprecated)]
    fn cause(&self) -> Option<&dyn Error> {
        Error::cause(&**self)
    }

    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Error::source(&**self)
    }

    fn backtrace(&self) -> Option<&Backtrace> {
        Error::backtrace(&**self)
    }
}

#[stable(feature = "fmt_error", since = "1.11.0")]
impl Error for fmt::Error {
    #[allow(deprecated)]
    fn description(&self) -> &str {
        "an error occurred when formatting an argument"
    }
}

#[stable(feature = "try_borrow", since = "1.13.0")]
impl Error for cell::BorrowError {
    #[allow(deprecated)]
    fn description(&self) -> &str {
        "already mutably borrowed"
    }
}

#[stable(feature = "try_borrow", since = "1.13.0")]
impl Error for cell::BorrowMutError {
    #[allow(deprecated)]
    fn description(&self) -> &str {
        "already borrowed"
    }
}

#[stable(feature = "try_from", since = "1.34.0")]
impl Error for char::CharTryFromError {
    #[allow(deprecated)]
    fn description(&self) -> &str {
        "converted integer out of range for `char`"
    }
}

#[stable(feature = "char_from_str", since = "1.20.0")]
impl Error for char::ParseCharError {
    #[allow(deprecated)]
    fn description(&self) -> &str {
        self.__description()
    }
}

#[stable(feature = "try_reserve", since = "1.57.0")]
impl Error for alloc::collections::TryReserveError {}

#[unstable(feature = "duration_checked_float", issue = "83400")]
impl Error for time::FromFloatSecsError {}

#[stable(feature = "rust1", since = "1.0.0")]
impl Error for alloc::ffi::NulError {
    #[allow(deprecated)]
    fn description(&self) -> &str {
        "nul byte found in data"
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl From<alloc::ffi::NulError> for io::Error {
    /// Converts a [`alloc::ffi::NulError`] into a [`io::Error`].
    fn from(_: alloc::ffi::NulError) -> io::Error {
        io::const_io_error!(io::ErrorKind::InvalidInput, "data provided contains a nul byte")
    }
}

#[stable(feature = "frombyteswithnulerror_impls", since = "1.17.0")]
impl Error for core::ffi::FromBytesWithNulError {
    #[allow(deprecated)]
    fn description(&self) -> &str {
        self.__description()
    }
}

#[unstable(feature = "cstr_from_bytes_until_nul", issue = "95027")]
impl Error for core::ffi::FromBytesUntilNulError {}

#[stable(feature = "cstring_from_vec_with_nul", since = "1.58.0")]
impl Error for alloc::ffi::FromVecWithNulError {}

#[stable(feature = "cstring_into", since = "1.7.0")]
impl Error for alloc::ffi::IntoStringError {
    #[allow(deprecated)]
    fn description(&self) -> &str {
        "C string contained non-utf8 bytes"
    }

    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(self.__source())
    }
}

// Copied from `any.rs`.
impl dyn Error + 'static {
    /// Returns `true` if the inner type is the same as `T`.
    #[stable(feature = "error_downcast", since = "1.3.0")]
    #[inline]
    pub fn is<T: Error + 'static>(&self) -> bool {
        // Get `TypeId` of the type this function is instantiated with.
        let t = TypeId::of::<T>();

        // Get `TypeId` of the type in the trait object (`self`).
        let concrete = self.type_id(private::Internal);

        // Compare both `TypeId`s on equality.
        t == concrete
    }

    /// Returns some reference to the inner value if it is of type `T`, or
    /// `None` if it isn't.
    #[stable(feature = "error_downcast", since = "1.3.0")]
    #[inline]
    pub fn downcast_ref<T: Error + 'static>(&self) -> Option<&T> {
        if self.is::<T>() {
            unsafe { Some(&*(self as *const dyn Error as *const T)) }
        } else {
            None
        }
    }

    /// Returns some mutable reference to the inner value if it is of type `T`, or
    /// `None` if it isn't.
    #[stable(feature = "error_downcast", since = "1.3.0")]
    #[inline]
    pub fn downcast_mut<T: Error + 'static>(&mut self) -> Option<&mut T> {
        if self.is::<T>() {
            unsafe { Some(&mut *(self as *mut dyn Error as *mut T)) }
        } else {
            None
        }
    }

    /// Request a reference to context of type `T`.
    #[unstable(feature = "generic_member_access", issue = "none")]
    pub fn request_ref<T: ?Sized + 'static>(&self) -> Option<&T> {
        core::any::request_ref(self)
    }

    /// Request a value to context of type `T`.
    #[unstable(feature = "generic_member_access", issue = "none")]
    pub fn request_value<T: 'static>(&self) -> Option<T> {
        core::any::request_value(self)
    }
}

impl dyn Error + 'static + Send {
    /// Forwards to the method defined on the type `dyn Error`.
    #[stable(feature = "error_downcast", since = "1.3.0")]
    #[inline]
    pub fn is<T: Error + 'static>(&self) -> bool {
        <dyn Error + 'static>::is::<T>(self)
    }

    /// Forwards to the method defined on the type `dyn Error`.
    #[stable(feature = "error_downcast", since = "1.3.0")]
    #[inline]
    pub fn downcast_ref<T: Error + 'static>(&self) -> Option<&T> {
        <dyn Error + 'static>::downcast_ref::<T>(self)
    }

    /// Forwards to the method defined on the type `dyn Error`.
    #[stable(feature = "error_downcast", since = "1.3.0")]
    #[inline]
    pub fn downcast_mut<T: Error + 'static>(&mut self) -> Option<&mut T> {
        <dyn Error + 'static>::downcast_mut::<T>(self)
    }

    /// Request a reference to context of type `T`.
    #[unstable(feature = "generic_member_access", issue = "none")]
    pub fn request_ref<T: ?Sized + 'static>(&self) -> Option<&T> {
        <dyn Error + 'static>::request_ref(self)
    }

    /// Request a value to context of type `T`.
    #[unstable(feature = "generic_member_access", issue = "none")]
    pub fn request_value<T: 'static>(&self) -> Option<T> {
        <dyn Error + 'static>::request_value(self)
    }
}

impl dyn Error + 'static + Send + Sync {
    /// Forwards to the method defined on the type `dyn Error`.
    #[stable(feature = "error_downcast", since = "1.3.0")]
    #[inline]
    pub fn is<T: Error + 'static>(&self) -> bool {
        <dyn Error + 'static>::is::<T>(self)
    }

    /// Forwards to the method defined on the type `dyn Error`.
    #[stable(feature = "error_downcast", since = "1.3.0")]
    #[inline]
    pub fn downcast_ref<T: Error + 'static>(&self) -> Option<&T> {
        <dyn Error + 'static>::downcast_ref::<T>(self)
    }

    /// Forwards to the method defined on the type `dyn Error`.
    #[stable(feature = "error_downcast", since = "1.3.0")]
    #[inline]
    pub fn downcast_mut<T: Error + 'static>(&mut self) -> Option<&mut T> {
        <dyn Error + 'static>::downcast_mut::<T>(self)
    }

    /// Request a reference to context of type `T`.
    #[unstable(feature = "generic_member_access", issue = "none")]
    pub fn request_ref<T: ?Sized + 'static>(&self) -> Option<&T> {
        <dyn Error + 'static>::request_ref(self)
    }

    /// Request a value to context of type `T`.
    #[unstable(feature = "generic_member_access", issue = "none")]
    pub fn request_value<T: 'static>(&self) -> Option<T> {
        <dyn Error + 'static>::request_value(self)
    }
}

impl dyn Error {
    #[inline]
    #[stable(feature = "error_downcast", since = "1.3.0")]
    /// Attempts to downcast the box to a concrete type.
    pub fn downcast<T: Error + 'static>(self: Box<Self>) -> Result<Box<T>, Box<dyn Error>> {
        if self.is::<T>() {
            unsafe {
                let raw: *mut dyn Error = Box::into_raw(self);
                Ok(Box::from_raw(raw as *mut T))
            }
        } else {
            Err(self)
        }
    }

    /// Returns an iterator starting with the current error and continuing with
    /// recursively calling [`Error::source`].
    ///
    /// If you want to omit the current error and only use its sources,
    /// use `skip(1)`.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(error_iter)]
    /// use std::error::Error;
    /// use std::fmt;
    ///
    /// #[derive(Debug)]
    /// struct A;
    ///
    /// #[derive(Debug)]
    /// struct B(Option<Box<dyn Error + 'static>>);
    ///
    /// impl fmt::Display for A {
    ///     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    ///         write!(f, "A")
    ///     }
    /// }
    ///
    /// impl fmt::Display for B {
    ///     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    ///         write!(f, "B")
    ///     }
    /// }
    ///
    /// impl Error for A {}
    ///
    /// impl Error for B {
    ///     fn source(&self) -> Option<&(dyn Error + 'static)> {
    ///         self.0.as_ref().map(|e| e.as_ref())
    ///     }
    /// }
    ///
    /// let b = B(Some(Box::new(A)));
    ///
    /// // let err : Box<Error> = b.into(); // or
    /// let err = &b as &(dyn Error);
    ///
    /// let mut iter = err.chain();
    ///
    /// assert_eq!("B".to_string(), iter.next().unwrap().to_string());
    /// assert_eq!("A".to_string(), iter.next().unwrap().to_string());
    /// assert!(iter.next().is_none());
    /// assert!(iter.next().is_none());
    /// ```
    #[unstable(feature = "error_iter", issue = "58520")]
    #[inline]
    pub fn chain(&self) -> Chain<'_> {
        Chain { current: Some(self) }
    }
}

/// An iterator over an [`Error`] and its sources.
///
/// If you want to omit the initial error and only process
/// its sources, use `skip(1)`.
#[unstable(feature = "error_iter", issue = "58520")]
#[derive(Clone, Debug)]
pub struct Chain<'a> {
    current: Option<&'a (dyn Error + 'static)>,
}

#[unstable(feature = "error_iter", issue = "58520")]
impl<'a> Iterator for Chain<'a> {
    type Item = &'a (dyn Error + 'static);

    fn next(&mut self) -> Option<Self::Item> {
        let current = self.current;
        self.current = self.current.and_then(Error::source);
        current
    }
}

impl dyn Error + Send {
    #[inline]
    #[stable(feature = "error_downcast", since = "1.3.0")]
    /// Attempts to downcast the box to a concrete type.
    pub fn downcast<T: Error + 'static>(self: Box<Self>) -> Result<Box<T>, Box<dyn Error + Send>> {
        let err: Box<dyn Error> = self;
        <dyn Error>::downcast(err).map_err(|s| unsafe {
            // Reapply the `Send` marker.
            transmute::<Box<dyn Error>, Box<dyn Error + Send>>(s)
        })
    }
}

impl dyn Error + Send + Sync {
    #[inline]
    #[stable(feature = "error_downcast", since = "1.3.0")]
    /// Attempts to downcast the box to a concrete type.
    pub fn downcast<T: Error + 'static>(self: Box<Self>) -> Result<Box<T>, Box<Self>> {
        let err: Box<dyn Error> = self;
        <dyn Error>::downcast(err).map_err(|s| unsafe {
            // Reapply the `Send + Sync` marker.
            transmute::<Box<dyn Error>, Box<dyn Error + Send + Sync>>(s)
        })
    }
}

/// An error reporter that prints an error and its sources.
///
/// Report also exposes configuration options for formatting the error chain, either entirely on a
/// single line, or in multi-line format with each cause in the error chain on a new line.
///
/// `Report` only requires that the wrapped error implement `Error`. It doesn't require that the
/// wrapped error be `Send`, `Sync`, or `'static`.
///
/// # Examples
///
/// ```rust
/// #![feature(error_reporter)]
/// use std::error::{Error, Report};
/// use std::fmt;
///
/// #[derive(Debug)]
/// struct SuperError {
///     source: SuperErrorSideKick,
/// }
///
/// impl fmt::Display for SuperError {
///     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
///         write!(f, "SuperError is here!")
///     }
/// }
///
/// impl Error for SuperError {
///     fn source(&self) -> Option<&(dyn Error + 'static)> {
///         Some(&self.source)
///     }
/// }
///
/// #[derive(Debug)]
/// struct SuperErrorSideKick;
///
/// impl fmt::Display for SuperErrorSideKick {
///     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
///         write!(f, "SuperErrorSideKick is here!")
///     }
/// }
///
/// impl Error for SuperErrorSideKick {}
///
/// fn get_super_error() -> Result<(), SuperError> {
///     Err(SuperError { source: SuperErrorSideKick })
/// }
///
/// fn main() {
///     match get_super_error() {
///         Err(e) => println!("Error: {}", Report::new(e)),
///         _ => println!("No error"),
///     }
/// }
/// ```
///
/// This example produces the following output:
///
/// ```console
/// Error: SuperError is here!: SuperErrorSideKick is here!
/// ```
///
/// ## Output consistency
///
/// Report prints the same output via `Display` and `Debug`, so it works well with
/// [`Result::unwrap`]/[`Result::expect`] which print their `Err` variant via `Debug`:
///
/// ```should_panic
/// #![feature(error_reporter)]
/// use std::error::Report;
/// # use std::error::Error;
/// # use std::fmt;
/// # #[derive(Debug)]
/// # struct SuperError {
/// #     source: SuperErrorSideKick,
/// # }
/// # impl fmt::Display for SuperError {
/// #     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
/// #         write!(f, "SuperError is here!")
/// #     }
/// # }
/// # impl Error for SuperError {
/// #     fn source(&self) -> Option<&(dyn Error + 'static)> {
/// #         Some(&self.source)
/// #     }
/// # }
/// # #[derive(Debug)]
/// # struct SuperErrorSideKick;
/// # impl fmt::Display for SuperErrorSideKick {
/// #     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
/// #         write!(f, "SuperErrorSideKick is here!")
/// #     }
/// # }
/// # impl Error for SuperErrorSideKick {}
/// # fn get_super_error() -> Result<(), SuperError> {
/// #     Err(SuperError { source: SuperErrorSideKick })
/// # }
///
/// get_super_error().map_err(Report::new).unwrap();
/// ```
///
/// This example produces the following output:
///
/// ```console
/// thread 'main' panicked at 'called `Result::unwrap()` on an `Err` value: SuperError is here!: SuperErrorSideKick is here!', src/error.rs:34:40
/// note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
/// ```
///
/// ## Return from `main`
///
/// `Report` also implements `From` for all types that implement [`Error`]; this when combined with
/// the `Debug` output means `Report` is an ideal starting place for formatting errors returned
/// from `main`.
///
/// ```should_panic
/// #![feature(error_reporter)]
/// use std::error::Report;
/// # use std::error::Error;
/// # use std::fmt;
/// # #[derive(Debug)]
/// # struct SuperError {
/// #     source: SuperErrorSideKick,
/// # }
/// # impl fmt::Display for SuperError {
/// #     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
/// #         write!(f, "SuperError is here!")
/// #     }
/// # }
/// # impl Error for SuperError {
/// #     fn source(&self) -> Option<&(dyn Error + 'static)> {
/// #         Some(&self.source)
/// #     }
/// # }
/// # #[derive(Debug)]
/// # struct SuperErrorSideKick;
/// # impl fmt::Display for SuperErrorSideKick {
/// #     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
/// #         write!(f, "SuperErrorSideKick is here!")
/// #     }
/// # }
/// # impl Error for SuperErrorSideKick {}
/// # fn get_super_error() -> Result<(), SuperError> {
/// #     Err(SuperError { source: SuperErrorSideKick })
/// # }
///
/// fn main() -> Result<(), Report> {
///     get_super_error()?;
///     Ok(())
/// }
/// ```
///
/// This example produces the following output:
///
/// ```console
/// Error: SuperError is here!: SuperErrorSideKick is here!
/// ```
///
/// **Note**: `Report`s constructed via `?` and `From` will be configured to use the single line
/// output format. If you want to make sure your `Report`s are pretty printed and include backtrace
/// you will need to manually convert and enable those flags.
///
/// ```should_panic
/// #![feature(error_reporter)]
/// use std::error::Report;
/// # use std::error::Error;
/// # use std::fmt;
/// # #[derive(Debug)]
/// # struct SuperError {
/// #     source: SuperErrorSideKick,
/// # }
/// # impl fmt::Display for SuperError {
/// #     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
/// #         write!(f, "SuperError is here!")
/// #     }
/// # }
/// # impl Error for SuperError {
/// #     fn source(&self) -> Option<&(dyn Error + 'static)> {
/// #         Some(&self.source)
/// #     }
/// # }
/// # #[derive(Debug)]
/// # struct SuperErrorSideKick;
/// # impl fmt::Display for SuperErrorSideKick {
/// #     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
/// #         write!(f, "SuperErrorSideKick is here!")
/// #     }
/// # }
/// # impl Error for SuperErrorSideKick {}
/// # fn get_super_error() -> Result<(), SuperError> {
/// #     Err(SuperError { source: SuperErrorSideKick })
/// # }
///
/// fn main() -> Result<(), Report> {
///     get_super_error()
///         .map_err(Report::from)
///         .map_err(|r| r.pretty(true).show_backtrace(true))?;
///     Ok(())
/// }
/// ```
///
/// This example produces the following output:
///
/// ```console
/// Error: SuperError is here!
///
/// Caused by:
///       SuperErrorSideKick is here!
/// ```
#[unstable(feature = "error_reporter", issue = "90172")]
pub struct Report<E = Box<dyn Error>> {
    /// The error being reported.
    error: E,
    /// Whether a backtrace should be included as part of the report.
    show_backtrace: bool,
    /// Whether the report should be pretty-printed.
    pretty: bool,
}

impl<E> Report<E>
where
    Report<E>: From<E>,
{
    /// Create a new `Report` from an input error.
    #[unstable(feature = "error_reporter", issue = "90172")]
    pub fn new(error: E) -> Report<E> {
        Self::from(error)
    }
}

impl<E> Report<E> {
    /// Enable pretty-printing the report across multiple lines.
    ///
    /// # Examples
    ///
    /// ```rust
    /// #![feature(error_reporter)]
    /// use std::error::Report;
    /// # use std::error::Error;
    /// # use std::fmt;
    /// # #[derive(Debug)]
    /// # struct SuperError {
    /// #     source: SuperErrorSideKick,
    /// # }
    /// # impl fmt::Display for SuperError {
    /// #     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    /// #         write!(f, "SuperError is here!")
    /// #     }
    /// # }
    /// # impl Error for SuperError {
    /// #     fn source(&self) -> Option<&(dyn Error + 'static)> {
    /// #         Some(&self.source)
    /// #     }
    /// # }
    /// # #[derive(Debug)]
    /// # struct SuperErrorSideKick;
    /// # impl fmt::Display for SuperErrorSideKick {
    /// #     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    /// #         write!(f, "SuperErrorSideKick is here!")
    /// #     }
    /// # }
    /// # impl Error for SuperErrorSideKick {}
    ///
    /// let error = SuperError { source: SuperErrorSideKick };
    /// let report = Report::new(error).pretty(true);
    /// eprintln!("Error: {report:?}");
    /// ```
    ///
    /// This example produces the following output:
    ///
    /// ```console
    /// Error: SuperError is here!
    ///
    /// Caused by:
    ///       SuperErrorSideKick is here!
    /// ```
    ///
    /// When there are multiple source errors the causes will be numbered in order of iteration
    /// starting from the outermost error.
    ///
    /// ```rust
    /// #![feature(error_reporter)]
    /// use std::error::Report;
    /// # use std::error::Error;
    /// # use std::fmt;
    /// # #[derive(Debug)]
    /// # struct SuperError {
    /// #     source: SuperErrorSideKick,
    /// # }
    /// # impl fmt::Display for SuperError {
    /// #     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    /// #         write!(f, "SuperError is here!")
    /// #     }
    /// # }
    /// # impl Error for SuperError {
    /// #     fn source(&self) -> Option<&(dyn Error + 'static)> {
    /// #         Some(&self.source)
    /// #     }
    /// # }
    /// # #[derive(Debug)]
    /// # struct SuperErrorSideKick {
    /// #     source: SuperErrorSideKickSideKick,
    /// # }
    /// # impl fmt::Display for SuperErrorSideKick {
    /// #     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    /// #         write!(f, "SuperErrorSideKick is here!")
    /// #     }
    /// # }
    /// # impl Error for SuperErrorSideKick {
    /// #     fn source(&self) -> Option<&(dyn Error + 'static)> {
    /// #         Some(&self.source)
    /// #     }
    /// # }
    /// # #[derive(Debug)]
    /// # struct SuperErrorSideKickSideKick;
    /// # impl fmt::Display for SuperErrorSideKickSideKick {
    /// #     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    /// #         write!(f, "SuperErrorSideKickSideKick is here!")
    /// #     }
    /// # }
    /// # impl Error for SuperErrorSideKickSideKick { }
    ///
    /// let source = SuperErrorSideKickSideKick;
    /// let source = SuperErrorSideKick { source };
    /// let error = SuperError { source };
    /// let report = Report::new(error).pretty(true);
    /// eprintln!("Error: {report:?}");
    /// ```
    ///
    /// This example produces the following output:
    ///
    /// ```console
    /// Error: SuperError is here!
    ///
    /// Caused by:
    ///    0: SuperErrorSideKick is here!
    ///    1: SuperErrorSideKickSideKick is here!
    /// ```
    #[unstable(feature = "error_reporter", issue = "90172")]
    pub fn pretty(mut self, pretty: bool) -> Self {
        self.pretty = pretty;
        self
    }

    /// Display backtrace if available when using pretty output format.
    ///
    /// # Examples
    ///
    /// **Note**: Report will search for the first `Backtrace` it can find starting from the
    /// outermost error. In this example it will display the backtrace from the second error in the
    /// chain, `SuperErrorSideKick`.
    ///
    /// ```rust
    /// #![feature(error_reporter)]
    /// #![feature(backtrace)]
    /// # use std::error::Error;
    /// # use std::fmt;
    /// use std::error::Report;
    /// use std::backtrace::Backtrace;
    ///
    /// # #[derive(Debug)]
    /// # struct SuperError {
    /// #     source: SuperErrorSideKick,
    /// # }
    /// # impl fmt::Display for SuperError {
    /// #     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    /// #         write!(f, "SuperError is here!")
    /// #     }
    /// # }
    /// # impl Error for SuperError {
    /// #     fn source(&self) -> Option<&(dyn Error + 'static)> {
    /// #         Some(&self.source)
    /// #     }
    /// # }
    /// #[derive(Debug)]
    /// struct SuperErrorSideKick {
    ///     backtrace: Backtrace,
    /// }
    ///
    /// impl SuperErrorSideKick {
    ///     fn new() -> SuperErrorSideKick {
    ///         SuperErrorSideKick { backtrace: Backtrace::force_capture() }
    ///     }
    /// }
    ///
    /// impl Error for SuperErrorSideKick {
    ///     fn backtrace(&self) -> Option<&Backtrace> {
    ///         Some(&self.backtrace)
    ///     }
    /// }
    ///
    /// // The rest of the example is unchanged ...
    /// # impl fmt::Display for SuperErrorSideKick {
    /// #     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    /// #         write!(f, "SuperErrorSideKick is here!")
    /// #     }
    /// # }
    ///
    /// let source = SuperErrorSideKick::new();
    /// let error = SuperError { source };
    /// let report = Report::new(error).pretty(true).show_backtrace(true);
    /// eprintln!("Error: {report:?}");
    /// ```
    ///
    /// This example produces something similar to the following output:
    ///
    /// ```console
    /// Error: SuperError is here!
    ///
    /// Caused by:
    ///       SuperErrorSideKick is here!
    ///
    /// Stack backtrace:
    ///    0: rust_out::main::_doctest_main_src_error_rs_1158_0::SuperErrorSideKick::new
    ///    1: rust_out::main::_doctest_main_src_error_rs_1158_0
    ///    2: rust_out::main
    ///    3: core::ops::function::FnOnce::call_once
    ///    4: std::sys_common::backtrace::__rust_begin_short_backtrace
    ///    5: std::rt::lang_start::{{closure}}
    ///    6: std::panicking::try
    ///    7: std::rt::lang_start_internal
    ///    8: std::rt::lang_start
    ///    9: main
    ///   10: __libc_start_main
    ///   11: _start
    /// ```
    #[unstable(feature = "error_reporter", issue = "90172")]
    pub fn show_backtrace(mut self, show_backtrace: bool) -> Self {
        self.show_backtrace = show_backtrace;
        self
    }
}

impl<E> Report<E>
where
    E: Error,
{
    fn backtrace(&self) -> Option<&Backtrace> {
        // have to grab the backtrace on the first error directly since that error may not be
        // 'static
        let backtrace = self.error.backtrace();
        let backtrace = backtrace.or_else(|| {
            self.error
                .source()
                .map(|source| source.chain().find_map(|source| source.backtrace()))
                .flatten()
        });
        backtrace
    }

    /// Format the report as a single line.
    #[unstable(feature = "error_reporter", issue = "90172")]
    fn fmt_singleline(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.error)?;

        let sources = self.error.source().into_iter().flat_map(<dyn Error>::chain);

        for cause in sources {
            write!(f, ": {cause}")?;
        }

        Ok(())
    }

    /// Format the report as multiple lines, with each error cause on its own line.
    #[unstable(feature = "error_reporter", issue = "90172")]
    fn fmt_multiline(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let error = &self.error;

        write!(f, "{error}")?;

        if let Some(cause) = error.source() {
            write!(f, "\n\nCaused by:")?;

            let multiple = cause.source().is_some();

            for (ind, error) in cause.chain().enumerate() {
                writeln!(f)?;
                let mut indented = Indented { inner: f };
                if multiple {
                    write!(indented, "{ind: >4}: {error}")?;
                } else {
                    write!(indented, "      {error}")?;
                }
            }
        }

        if self.show_backtrace {
            let backtrace = self.backtrace();

            if let Some(backtrace) = backtrace {
                let backtrace = backtrace.to_string();

                f.write_str("\n\nStack backtrace:\n")?;
                f.write_str(backtrace.trim_end())?;
            }
        }

        Ok(())
    }
}

impl Report<Box<dyn Error>> {
    fn backtrace(&self) -> Option<&Backtrace> {
        // have to grab the backtrace on the first error directly since that error may not be
        // 'static
        let backtrace = self.error.backtrace();
        let backtrace = backtrace.or_else(|| {
            self.error
                .source()
                .map(|source| source.chain().find_map(|source| source.backtrace()))
                .flatten()
        });
        backtrace
    }

    /// Format the report as a single line.
    #[unstable(feature = "error_reporter", issue = "90172")]
    fn fmt_singleline(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.error)?;

        let sources = self.error.source().into_iter().flat_map(<dyn Error>::chain);

        for cause in sources {
            write!(f, ": {cause}")?;
        }

        Ok(())
    }

    /// Format the report as multiple lines, with each error cause on its own line.
    #[unstable(feature = "error_reporter", issue = "90172")]
    fn fmt_multiline(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let error = &self.error;

        write!(f, "{error}")?;

        if let Some(cause) = error.source() {
            write!(f, "\n\nCaused by:")?;

            let multiple = cause.source().is_some();

            for (ind, error) in cause.chain().enumerate() {
                writeln!(f)?;
                let mut indented = Indented { inner: f };
                if multiple {
                    write!(indented, "{ind: >4}: {error}")?;
                } else {
                    write!(indented, "      {error}")?;
                }
            }
        }

        if self.show_backtrace {
            let backtrace = self.backtrace();

            if let Some(backtrace) = backtrace {
                let backtrace = backtrace.to_string();

                f.write_str("\n\nStack backtrace:\n")?;
                f.write_str(backtrace.trim_end())?;
            }
        }

        Ok(())
    }
}

#[unstable(feature = "error_reporter", issue = "90172")]
impl<E> From<E> for Report<E>
where
    E: Error,
{
    fn from(error: E) -> Self {
        Report { error, show_backtrace: false, pretty: false }
    }
}

#[unstable(feature = "error_reporter", issue = "90172")]
impl<'a, E> From<E> for Report<Box<dyn Error + 'a>>
where
    E: Error + 'a,
{
    fn from(error: E) -> Self {
        let error = box error;
        Report { error, show_backtrace: false, pretty: false }
    }
}

#[unstable(feature = "error_reporter", issue = "90172")]
impl<E> fmt::Display for Report<E>
where
    E: Error,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.pretty { self.fmt_multiline(f) } else { self.fmt_singleline(f) }
    }
}

#[unstable(feature = "error_reporter", issue = "90172")]
impl fmt::Display for Report<Box<dyn Error>> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.pretty { self.fmt_multiline(f) } else { self.fmt_singleline(f) }
    }
}

// This type intentionally outputs the same format for `Display` and `Debug`for
// situations where you unwrap a `Report` or return it from main.
#[unstable(feature = "error_reporter", issue = "90172")]
impl<E> fmt::Debug for Report<E>
where
    Report<E>: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

/// Wrapper type for indenting the inner source.
struct Indented<'a, D> {
    inner: &'a mut D,
}

impl<T> Write for Indented<'_, T>
where
    T: Write,
{
    fn write_str(&mut self, s: &str) -> fmt::Result {
        for (i, line) in s.split('\n').enumerate() {
            if i > 0 {
                self.inner.write_char('\n')?;
                self.inner.write_str("      ")?;
            }

            self.inner.write_str(line)?;
        }

        Ok(())
    }
}
