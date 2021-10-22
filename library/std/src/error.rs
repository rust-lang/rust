//! Traits for working with Errors.

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
use crate::any::TypeId;
use crate::backtrace::Backtrace;
use crate::borrow::Cow;
use crate::cell;
use crate::char;
use crate::fmt::{self, Debug, Display, Write};
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
    ///     side: SuperErrorSideKick,
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
    ///         Some(&self.side)
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
    ///     Err(SuperError { side: SuperErrorSideKick })
    /// }
    ///
    /// fn main() {
    ///     match get_super_error() {
    ///         Err(e) => {
    ///             println!("Error: {}", e);
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
    ///     eprintln!("Error: {}", e);
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_deprecated(since = "1.42.0", reason = "use the Display impl or to_string()")]
    fn description(&self) -> &str {
        "description() is deprecated; use Display"
    }

    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_deprecated(
        since = "1.33.0",
        reason = "replaced by Error::source, which can support downcasting"
    )]
    #[allow(missing_docs)]
    fn cause(&self) -> Option<&dyn Error> {
        self.source()
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
impl Error for time::FromSecsError {}

// Copied from `any.rs`.
impl dyn Error + 'static {
    /// Returns `true` if the boxed type is the same as `T`
    #[stable(feature = "error_downcast", since = "1.3.0")]
    #[inline]
    pub fn is<T: Error + 'static>(&self) -> bool {
        // Get `TypeId` of the type this function is instantiated with.
        let t = TypeId::of::<T>();

        // Get `TypeId` of the type in the trait object.
        let boxed = self.type_id(private::Internal);

        // Compare both `TypeId`s on equality.
        t == boxed
    }

    /// Returns some reference to the boxed value if it is of type `T`, or
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

    /// Returns some mutable reference to the boxed value if it is of type `T`, or
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

/// An error reporter that exposes the entire error chain for printing.
/// It also exposes options for formatting the error chain, either entirely on a single line,
/// or in multi-line format with each cause in the error chain on a new line.
///
/// # Examples
///
/// ```
/// #![feature(error_reporter)]
///
/// use std::error::{Error, Report};
/// use std::fmt;
///
/// #[derive(Debug)]
/// struct SuperError {
///     side: SuperErrorSideKick,
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
///         Some(&self.side)
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
/// fn main() {
///     let error = SuperError { side: SuperErrorSideKick };
///     let report = Report::new(&error).pretty(true);
///
///     println!("{}", report);
/// }
/// ```
#[unstable(feature = "error_reporter", issue = "90172")]
pub struct Report<E> {
    source: E,
    show_backtrace: bool,
    pretty: bool,
}

impl<E> Report<E>
where
    E: Error,
{
    /// Create a new `Report` from an input error.
    #[unstable(feature = "error_reporter", issue = "90172")]
    pub fn new(source: E) -> Report<E> {
        Report { source, show_backtrace: false, pretty: false }
    }

    /// Enable pretty-printing the report.
    #[unstable(feature = "error_reporter", issue = "90172")]
    pub fn pretty(mut self, pretty: bool) -> Self {
        self.pretty = pretty;
        self
    }

    /// Enable showing a backtrace for the report.
    #[unstable(feature = "error_reporter", issue = "90172")]
    pub fn show_backtrace(mut self, show_backtrace: bool) -> Self {
        self.show_backtrace = show_backtrace;
        self
    }

    /// Format the report as a single line.
    #[unstable(feature = "error_reporter", issue = "90172")]
    fn fmt_singleline(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.source)?;

        let sources = self.source.source().into_iter().flat_map(<dyn Error>::chain);

        for cause in sources {
            write!(f, ": {}", cause)?;
        }

        Ok(())
    }

    /// Format the report as multiple lines, with each error cause on its own line.
    #[unstable(feature = "error_reporter", issue = "90172")]
    fn fmt_multiline(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let error = &self.source;

        write!(f, "{}", error)?;

        if let Some(cause) = error.source() {
            write!(f, "\n\nCaused by:")?;

            let multiple = cause.source().is_some();
            let format = if multiple {
                Format::Numbered { ind: 0 }
            } else {
                Format::Uniform { indentation: "    " }
            };

            for error in cause.chain() {
                writeln!(f)?;

                let mut indented = Indented { inner: f, needs_indent: true, format };

                write!(indented, "{}", error)?;
            }
        }

        if self.show_backtrace {
            let backtrace = error.backtrace();

            if let Some(backtrace) = backtrace {
                let mut backtrace = backtrace.to_string();

                write!(f, "\n\n")?;
                writeln!(f, "Stack backtrace:")?;

                backtrace.truncate(backtrace.trim_end().len());

                write!(f, "{}", backtrace)?;
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
    fn from(source: E) -> Self {
        Report::new(source)
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

// This type intentionally outputs the same format for `Display` and `Debug`for
// situations where you unwrap a `Report` or return it from main.
#[unstable(feature = "error_reporter", issue = "90172")]
impl<E> fmt::Debug for Report<E>
where
    E: Error,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

/// Encapsulates how error sources are indented and formatted.
struct Indented<'a, D: ?Sized> {
    inner: &'a mut D,
    needs_indent: bool,
    format: Format,
}

/// The possible variants that error sources can be formatted as.
#[derive(Clone, Copy)]
enum Format {
    /// Insert uniform indentation before every line.
    ///
    /// This format takes a static string as input and inserts it after every newline.
    Uniform {
        /// The string to insert as indentation.
        indentation: &'static str,
    },
    /// Inserts a number before the first line.
    ///
    /// This format hard codes the indentation level to match the indentation from
    /// `std::backtrace::Backtrace`.
    Numbered {
        /// The index to insert before the first line of output.
        ind: usize,
    },
}

impl<D> Write for Indented<'_, D>
where
    D: Write + ?Sized,
{
    fn write_str(&mut self, s: &str) -> fmt::Result {
        for (ind, line) in s.split('\n').enumerate() {
            if ind > 0 {
                self.inner.write_char('\n')?;
                self.needs_indent = true;
            }

            if self.needs_indent {
                if line.is_empty() {
                    continue;
                }

                self.format.insert_indentation(ind, &mut self.inner)?;
                self.needs_indent = false;
            }

            self.inner.write_fmt(format_args!("{}", line))?;
        }

        Ok(())
    }
}

impl Format {
    /// Write the specified formatting to the write buffer.
    fn insert_indentation(&mut self, line: usize, f: &mut dyn Write) -> fmt::Result {
        match self {
            Format::Uniform { indentation } => {
                write!(f, "{}", indentation)
            }
            Format::Numbered { ind } => {
                if line == 0 {
                    write!(f, "{: >4}: ", ind)?;
                    *ind += 1;
                    Ok(())
                } else {
                    write!(f, "      ")
                }
            }
        }
    }
}
