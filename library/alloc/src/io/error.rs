#[unstable(feature = "raw_os_error_ty", issue = "107792")]
pub use core::io::RawOsError;
#[unstable(feature = "io_const_error_internals", issue = "none")]
pub use core::io::SimpleMessage;
#[unstable(feature = "io_const_error", issue = "133448")]
pub use core::io::const_error;
#[unstable(feature = "core_io_internals", reason = "exposed only for libstd", issue = "none")]
pub use core::io::{Custom, CustomOwner, OsFunctions};
#[unstable(feature = "alloc_io", issue = "154046")]
pub use core::io::{Error, ErrorKind, Result};
use core::{error, result};

use crate::boxed::Box;

impl Error {
    /// Creates a new I/O error from a known kind of error as well as an
    /// arbitrary error payload.
    ///
    /// This function is used to generically create I/O errors which do not
    /// originate from the OS itself. The `error` argument is an arbitrary
    /// payload which will be contained in this [`Error`].
    ///
    /// Note that this function allocates memory on the heap.
    /// If no extra payload is required, use the `From` conversion from
    /// `ErrorKind`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::io::{Error, ErrorKind};
    ///
    /// // errors can be created from strings
    /// let custom_error = Error::new(ErrorKind::Other, "oh no!");
    ///
    /// // errors can also be created from other errors
    /// let custom_error2 = Error::new(ErrorKind::Interrupted, custom_error);
    ///
    /// // creating an error without payload (and without memory allocation)
    /// let eof_error = Error::from(ErrorKind::UnexpectedEof);
    /// ```
    #[cfg(not(no_global_oom_handling))]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[cfg_attr(not(test), rustc_diagnostic_item = "io_error_new")]
    #[inline(never)]
    #[rustc_allow_incoherent_impl]
    pub fn new<E>(kind: ErrorKind, error: E) -> Error
    where
        E: Into<Box<dyn error::Error + Send + Sync>>,
    {
        error_from_box(kind, error.into())
    }

    /// Creates a new I/O error from an arbitrary error payload.
    ///
    /// This function is used to generically create I/O errors which do not
    /// originate from the OS itself. It is a shortcut for [`Error::new`][new]
    /// with [`ErrorKind::Other`].
    ///
    /// [new]: struct.Error.html#method.new
    ///
    /// # Examples
    ///
    /// ```
    /// use std::io::Error;
    ///
    /// // errors can be created from strings
    /// let custom_error = Error::other("oh no!");
    ///
    /// // errors can also be created from other errors
    /// let custom_error2 = Error::other(custom_error);
    /// ```
    #[cfg(not(no_global_oom_handling))]
    #[stable(feature = "io_error_other", since = "1.74.0")]
    #[rustc_allow_incoherent_impl]
    pub fn other<E>(error: E) -> Error
    where
        E: Into<Box<dyn error::Error + Send + Sync>>,
    {
        error_from_box(ErrorKind::Other, error.into())
    }

    /// Consumes the `Error`, returning its inner error (if any).
    ///
    /// If this [`Error`] was constructed via [`new`][new] or [`other`][other],
    /// then this function will return [`Some`],
    /// otherwise it will return [`None`].
    ///
    /// [new]: struct.Error.html#method.new
    /// [other]: struct.Error.html#method.other
    ///
    /// # Examples
    ///
    /// ```
    /// use std::io::{Error, ErrorKind};
    ///
    /// fn print_error(err: Error) {
    ///     if let Some(inner_err) = err.into_inner() {
    ///         println!("Inner error: {inner_err}");
    ///     } else {
    ///         println!("No inner error");
    ///     }
    /// }
    ///
    /// fn main() {
    ///     // Will print "No inner error".
    ///     print_error(Error::last_os_error());
    ///     // Will print "Inner error: ...".
    ///     print_error(Error::new(ErrorKind::Other, "oh no!"));
    /// }
    /// ```
    #[stable(feature = "io_error_inner", since = "1.3.0")]
    #[must_use = "`self` will be dropped if the result is not used"]
    #[inline]
    #[rustc_allow_incoherent_impl]
    pub fn into_inner(self) -> Option<Box<dyn error::Error + Send + Sync>> {
        let custom_owner = self.into_custom_owner().ok()?;

        let ptr = custom_owner.into_raw().as_ptr();

        // SAFETY:
        // `Error` can only contain a `CustomOwner` if it was constructed using `Box::into_raw`.
        let custom = unsafe { Box::<Custom>::from_raw(ptr) };

        let ptr = custom.into_raw().as_ptr();

        // SAFETY:
        // Any `CustomOwner` from an `Error` was constructed by the `alloc` crate
        // to contain a `Custom` which itself was constructed with `Box::into_raw`.
        Some(unsafe { Box::from_raw(ptr) })
    }

    /// Attempts to downcast the custom boxed error to `E`.
    ///
    /// If this [`Error`] contains a custom boxed error,
    /// then it would attempt downcasting on the boxed error,
    /// otherwise it will return [`Err`].
    ///
    /// If the custom boxed error has the same type as `E`, it will return [`Ok`],
    /// otherwise it will also return [`Err`].
    ///
    /// This method is meant to be a convenience routine for calling
    /// `Box<dyn Error + Sync + Send>::downcast` on the custom boxed error, returned by
    /// [`Error::into_inner`][into_inner].
    ///
    /// [into_inner]: struct.Error.html#method.into_inner
    ///
    /// # Examples
    ///
    /// ```
    /// use std::fmt;
    /// use std::io;
    /// use std::error::Error;
    ///
    /// #[derive(Debug)]
    /// enum E {
    ///     Io(io::Error),
    ///     SomeOtherVariant,
    /// }
    ///
    /// impl fmt::Display for E {
    ///    // ...
    /// #    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    /// #        todo!()
    /// #    }
    /// }
    /// impl Error for E {}
    ///
    /// impl From<io::Error> for E {
    ///     fn from(err: io::Error) -> E {
    ///         err.downcast::<E>()
    ///             .unwrap_or_else(E::Io)
    ///     }
    /// }
    ///
    /// impl From<E> for io::Error {
    ///     fn from(err: E) -> io::Error {
    ///         match err {
    ///             E::Io(io_error) => io_error,
    ///             e => io::Error::new(io::ErrorKind::Other, e),
    ///         }
    ///     }
    /// }
    ///
    /// # fn main() {
    /// let e = E::SomeOtherVariant;
    /// // Convert it to an io::Error
    /// let io_error = io::Error::from(e);
    /// // Cast it back to the original variant
    /// let e = E::from(io_error);
    /// assert!(matches!(e, E::SomeOtherVariant));
    ///
    /// let io_error = io::Error::from(io::ErrorKind::AlreadyExists);
    /// // Convert it to E
    /// let e = E::from(io_error);
    /// // Cast it back to the original variant
    /// let io_error = io::Error::from(e);
    /// assert_eq!(io_error.kind(), io::ErrorKind::AlreadyExists);
    /// assert!(io_error.get_ref().is_none());
    /// assert!(io_error.raw_os_error().is_none());
    /// # }
    /// ```
    #[stable(feature = "io_error_downcast", since = "1.79.0")]
    #[rustc_allow_incoherent_impl]
    pub fn downcast<E>(self) -> result::Result<E, Self>
    where
        E: error::Error + Send + Sync + 'static,
    {
        if let Some(e) = self.get_ref()
            && e.is::<E>()
        {
            if let Some(b) = self.into_inner()
                && let Ok(err) = b.downcast::<E>()
            {
                Ok(*err)
            } else {
                // Safety: We have just checked that the condition is true
                unsafe { core::hint::unreachable_unchecked() }
            }
        } else {
            Err(self)
        }
    }
}

#[cfg(all(not(no_rc), not(no_sync), not(no_global_oom_handling)))]
#[stable(feature = "rust1", since = "1.0.0")]
impl From<crate::ffi::NulError> for Error {
    /// Converts a [`crate::ffi::NulError`] into a [`Error`].
    fn from(_: crate::ffi::NulError) -> Error {
        const_error!(ErrorKind::InvalidInput, "data provided contains a nul byte")
    }
}

#[stable(feature = "io_error_from_try_reserve", since = "1.78.0")]
impl From<crate::collections::TryReserveError> for Error {
    /// Converts `TryReserveError` to an error with [`ErrorKind::OutOfMemory`].
    ///
    /// `TryReserveError` won't be available as the error `source()`,
    /// but this may change in the future.
    fn from(_: crate::collections::TryReserveError) -> Error {
        // ErrorData::Custom allocates, which isn't great for handling OOM errors.
        ErrorKind::OutOfMemory.into()
    }
}

#[cfg(not(no_global_oom_handling))]
fn error_from_box(kind: ErrorKind, error: Box<dyn error::Error + Send + Sync>) -> Error {
    /// # Safety
    ///
    /// `ptr` must be valid to pass into `Box::from_raw`.
    unsafe fn drop_box_raw<T: ?Sized>(ptr: *mut T) {
        // SAFETY
        // Caller ensures `ptr` is valid to pass into `Box::from_raw`.
        drop(unsafe { Box::from_raw(ptr) })
    }

    // SAFETY: the pointer returned by Box::into_raw is non-null.
    let error = unsafe { core::ptr::NonNull::new_unchecked(Box::into_raw(error)) };

    // SAFETY:
    // * `error` is valid up to a static lifetime, and owns its pointee.
    // * `drop_box_raw` is safe to call for the pointer `error` exactly once.
    // * `drop_box_raw` is safe to call on a pointer to this instance of `Custom`,
    //   and will be stored in a `CustomOwner`.
    let custom = unsafe { Custom::from_raw(kind, error, drop_box_raw, drop_box_raw) };

    // SAFETY: the pointer returned by Box::into_raw is non-null.
    let custom = unsafe { core::ptr::NonNull::new_unchecked(Box::into_raw(Box::new(custom))) };

    // SAFETY: the `outer_drop` provided to `custom` is valid for itself.
    let custom_owner = unsafe { CustomOwner::from_raw(custom) };

    // SAFETY: `custom_owner` has bee constructed from a `Box` from the `alloc` crate.
    unsafe { Error::from_custom_owner(custom_owner) }
}
