#![doc = include_str!("../error.md")]
#![stable(feature = "error_in_core", since = "1.81.0")]

use crate::any::TypeId;
use crate::fmt::{Debug, Display};

#[unstable(feature = "error_generic_member_multi_access", issue = "149615")]
pub mod provide;

#[unstable(feature = "error_generic_member_access", issue = "99301")]
pub use provide::{Request, request_ref, request_value};

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
/// # Error source
///
/// Errors may provide cause information. [`Error::source()`] is generally
/// used when errors cross "abstraction boundaries". If one module must report
/// an error that is caused by an error from a lower-level module, it can allow
/// accessing that error via `Error::source()`. This makes it possible for the
/// high-level module to provide its own errors while also revealing some of the
/// implementation for debugging.
///
/// In error types that wrap an underlying error, the underlying error
/// should be either returned by the outer error's `Error::source()`, or rendered
/// by the outer error's `Display` implementation, but not both.
///
/// # Example
///
/// Implementing the `Error` trait only requires that `Debug` and `Display` are implemented too.
///
/// ```
/// use std::error::Error;
/// use std::fmt;
/// use std::path::PathBuf;
///
/// #[derive(Debug)]
/// struct ReadConfigError {
///     path: PathBuf
/// }
///
/// impl fmt::Display for ReadConfigError {
///     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
///         let path = self.path.display();
///         write!(f, "unable to read configuration at {path}")
///     }
/// }
///
/// impl Error for ReadConfigError {}
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_diagnostic_item = "Error"]
#[rustc_has_incoherent_inherent_impls]
#[allow(multiple_supertrait_upcastable)]
pub trait Error: Debug + Display {
    /// Returns the lower-level source of this error, if any.
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

    /// Provides type-based access to context intended for error reports.
    ///
    /// Used in conjunction with [`Request::provide_value`] and [`Request::provide_ref`] to extract
    /// references to member variables from `dyn Error` trait objects.
    ///
    /// Every individual error type can `provide` some types that contain programmatic
    /// information about the error. The set of values `provide`d by a given error type
    /// can normally change between different versions of that error type's library. However,
    /// a library defining an error type can always make stronger backwards-compatibility
    /// promises - for example, a library can declare that an error type always provides a
    /// [`Location`](core::panic::Location) that provides a relevant source-code location.
    ///
    /// # Whether to provide by reference or by value
    ///
    /// [`Request::provide_value`] and [`Request::provide_ref`] are two different namespaces.
    /// Therefore, when providing a type, it needs to be picked whether it will be provided
    /// by reference or by value.
    ///
    /// If a type is provided by value, then a new copy of that type has to be created every
    /// time it is provided, but if it is provided by reference, then the provided value has
    /// to be stored somewhere within the error so that the reference can be returned.
    ///
    /// Some general rules:
    ///
    /// 1. If a type is [Copy], it is conventional to provide it by value.
    /// 2. If a type is not [Copy] but also not computed at provide time, for example
    ///    backtrace types that are captured when the error is created, it is conventional
    ///    to provide it by reference.
    ///
    /// Provided types that are not [Copy] and computed at provide time are fairly rare in
    /// practice. However, when using them, you should be using
    /// [`Request::would_be_satisfied_by_value_of`] to avoid computing them when they
    /// are not requested.
    ///
    /// # Common uses of `provide`
    ///
    /// 1. [`Location`](core::panic::Location), provided by value, to indicate a source-code
    ///    location relevant to the error. This allows following the [`Error::source`]
    ///    chain to generate a "logical" backtrace, even in the absence of debug information.
    /// 2. A backtrace, provided by reference, that contains the backtrace of the error.
    /// 3. Various exit code types, normally provided by value. For example, an HTTP framework
    ///    might request an HTTP status on an error, to allow error types to override the HTTP
    ///    status returned on an error (consult your framework for specific behavior).
    ///
    /// # Example
    ///
    /// ```rust
    /// #![feature(error_generic_member_access)]
    /// use core::fmt;
    /// use core::error::{request_ref, request_value, Request};
    /// use core::panic::Location;
    ///
    /// #[derive(Debug)]
    /// enum MyLittleTeaPot {
    ///     Empty,
    /// }
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
    /// struct Error {
    ///     backtrace: MyBacktrace,
    ///     location: Location<'static>,
    /// }
    ///
    /// impl fmt::Display for Error {
    ///     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    ///         write!(f, "Example Error")
    ///     }
    /// }
    ///
    /// impl std::error::Error for Error {
    ///     fn provide<'a>(&'a self, request: &mut Request<'a>) {
    ///         request
    ///             .provide_ref::<MyBacktrace>(&self.backtrace)
    ///             .provide_value::<Location<'_>>(self.location);
    ///     }
    /// }
    ///
    /// fn main() {
    ///     let backtrace = MyBacktrace::new();
    ///     let location = Location::caller();
    ///     let error = Error { backtrace, location: *location };
    ///     let dyn_error = &error as &dyn std::error::Error;
    ///     let backtrace_ref = request_ref::<MyBacktrace>(dyn_error).unwrap();
    ///     let location = request_value::<Location<'_>>(dyn_error).unwrap();
    ///
    ///     assert!(core::ptr::eq(&error.backtrace, backtrace_ref));
    ///     assert_eq!(error.location, location);
    ///     assert!(request_ref::<MyLittleTeaPot>(dyn_error).is_none());
    /// }
    /// ```
    ///
    /// # Delegating Impls
    ///
    /// <div class="warning">
    ///
    /// **Warning**: We recommend implementors avoid delegating implementations of `provide` to
    /// source error implementations.
    ///
    /// </div>
    ///
    /// This method should expose context from the current piece of the source chain only, not from
    /// sources that are exposed in the chain of sources. Delegating `provide` implementations cause
    /// the same context to be provided by multiple errors in the chain of sources which can cause
    /// unintended duplication of information in error reports or require heuristics to deduplicate.
    ///
    /// In other words, the following implementation pattern for `provide` is discouraged and should
    /// not be used for [`Error`] types exposed in public APIs to third parties.
    ///
    /// ```rust
    /// # #![feature(error_generic_member_access)]
    /// # use core::fmt;
    /// # use core::error::Request;
    /// # #[derive(Debug)]
    /// struct MyError {
    ///     source: Error,
    /// }
    /// # #[derive(Debug)]
    /// # struct Error;
    /// # impl fmt::Display for Error {
    /// #     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    /// #         write!(f, "Example Source Error")
    /// #     }
    /// # }
    /// # impl fmt::Display for MyError {
    /// #     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    /// #         write!(f, "Example Error")
    /// #     }
    /// # }
    /// # impl std::error::Error for Error { }
    ///
    /// impl std::error::Error for MyError {
    ///     fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
    ///         Some(&self.source)
    ///     }
    ///
    ///     fn provide<'a>(&'a self, request: &mut Request<'a>) {
    ///         self.source.provide(request) // <--- Discouraged
    ///     }
    /// }
    /// ```
    #[unstable(feature = "error_generic_member_access", issue = "99301")]
    #[allow(unused_variables)]
    fn provide<'a>(&'a self, request: &mut Request<'a>) {}
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
    /// let err = &b as &dyn Error;
    ///
    /// let mut iter = err.sources();
    ///
    /// assert_eq!("B".to_string(), iter.next().unwrap().to_string());
    /// assert_eq!("A".to_string(), iter.next().unwrap().to_string());
    /// assert!(iter.next().is_none());
    /// assert!(iter.next().is_none());
    /// ```
    #[unstable(feature = "error_iter", issue = "58520")]
    #[inline]
    pub fn sources(&self) -> Source<'_> {
        // You may think this method would be better in the `Error` trait, and you'd be right.
        // Unfortunately that doesn't work, not because of the dyn-incompatibility rules but
        // because we save a reference to `self` in `Source`s below as a trait object.
        // If this method was declared in `Error`, then `self` would have the type `&T` where
        // `T` is some concrete type which implements `Error`. We would need to coerce `self`
        // to have type `&dyn Error`, but that requires that `Self` has a known size
        // (i.e., `Self: Sized`). We can't put that bound on `Error` since that would forbid
        // `Error` trait objects, and we can't put that bound on the method because that means
        // the method can't be called on trait objects (we'd also need the `'static` bound,
        // but that isn't allowed because methods with bounds on `Self` other than `Sized` are
        // dyn-incompatible). Requiring an `Unsize` bound is not backwards compatible.

        Source { current: Some(self) }
    }
}

/// An iterator over an [`Error`] and its sources.
///
/// If you want to omit the initial error and only process
/// its sources, use `skip(1)`.
#[unstable(feature = "error_iter", issue = "58520")]
#[derive(Clone, Debug)]
pub struct Source<'a> {
    current: Option<&'a (dyn Error + 'static)>,
}

#[unstable(feature = "error_iter", issue = "58520")]
impl<'a> Iterator for Source<'a> {
    type Item = &'a (dyn Error + 'static);

    fn next(&mut self) -> Option<Self::Item> {
        let current = self.current;
        self.current = self.current.and_then(Error::source);
        current
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.current.is_some() { (1, None) } else { (0, Some(0)) }
    }
}

#[unstable(feature = "error_iter", issue = "58520")]
impl<'a> crate::iter::FusedIterator for Source<'a> {}

#[stable(feature = "error_by_ref", since = "1.51.0")]
impl<'a, T: Error + ?Sized> Error for &'a T {
    #[allow(deprecated)]
    fn cause(&self) -> Option<&dyn Error> {
        Error::cause(&**self)
    }

    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Error::source(&**self)
    }

    fn provide<'b>(&'b self, request: &mut Request<'b>) {
        Error::provide(&**self, request);
    }
}

#[stable(feature = "fmt_error", since = "1.11.0")]
impl Error for crate::fmt::Error {}

#[stable(feature = "try_borrow", since = "1.13.0")]
impl Error for crate::cell::BorrowError {}

#[stable(feature = "try_borrow", since = "1.13.0")]
impl Error for crate::cell::BorrowMutError {}

#[stable(feature = "try_from", since = "1.34.0")]
impl Error for crate::char::CharTryFromError {}

#[stable(feature = "duration_checked_float", since = "1.66.0")]
impl Error for crate::time::TryFromFloatSecsError {}

#[stable(feature = "cstr_from_bytes_until_nul", since = "1.69.0")]
impl Error for crate::ffi::FromBytesUntilNulError {}

#[stable(feature = "get_many_mut", since = "1.86.0")]
impl Error for crate::slice::GetDisjointMutError {}
