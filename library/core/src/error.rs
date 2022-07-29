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

use crate::any::{Demand, Provider, TypeId};
use crate::fmt::{Debug, Display};

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
#[rustc_has_incoherent_inherent_impls]
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
    /// Used in conjunction with [`Demand::provide_value`] and [`Demand::provide_ref`] to extract
    /// references to member variables from `dyn Error` trait objects.
    ///
    /// # Example
    ///
    /// ```rust
    /// #![feature(provide_any)]
    /// #![feature(error_generic_member_access)]
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
    #[unstable(feature = "error_generic_member_access", issue = "99301")]
    #[allow(unused_variables)]
    fn provide<'a>(&'a self, req: &mut Demand<'a>) {}
}

#[unstable(feature = "error_generic_member_access", issue = "99301")]
impl<'b> Provider for dyn Error + 'b {
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

#[unstable(feature = "never_type", issue = "35121")]
impl Error for ! {}

impl<'a> dyn Error + 'a {
    /// Request a reference of type `T` as context about this error.
    #[unstable(feature = "error_generic_member_access", issue = "99301")]
    pub fn request_ref<T: ?Sized + 'static>(&'a self) -> Option<&'a T> {
        core::any::request_ref(self)
    }

    /// Request a value of type `T` as context about this error.
    #[unstable(feature = "error_generic_member_access", issue = "99301")]
    pub fn request_value<T: 'static>(&'a self) -> Option<T> {
        core::any::request_value(self)
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
            // SAFETY: `is` ensures this type cast is correct
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
            // SAFETY: `is` ensures this type cast is correct
            unsafe { Some(&mut *(self as *mut dyn Error as *mut T)) }
        } else {
            None
        }
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

    /// Request a reference of type `T` as context about this error.
    #[unstable(feature = "error_generic_member_access", issue = "99301")]
    pub fn request_ref<T: ?Sized + 'static>(&self) -> Option<&T> {
        <dyn Error>::request_ref(self)
    }

    /// Request a value of type `T` as context about this error.
    #[unstable(feature = "error_generic_member_access", issue = "99301")]
    pub fn request_value<T: 'static>(&self) -> Option<T> {
        <dyn Error>::request_value(self)
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

    /// Request a reference of type `T` as context about this error.
    #[unstable(feature = "error_generic_member_access", issue = "99301")]
    pub fn request_ref<T: ?Sized + 'static>(&self) -> Option<&T> {
        <dyn Error>::request_ref(self)
    }

    /// Request a value of type `T` as context about this error.
    #[unstable(feature = "error_generic_member_access", issue = "99301")]
    pub fn request_value<T: 'static>(&self) -> Option<T> {
        <dyn Error>::request_value(self)
    }
}

impl dyn Error {
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
