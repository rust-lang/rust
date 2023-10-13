use crate::{alloc::Global, boxed::Box, string::String};
use core::io::{
    const_io_error,
    error_internals::{AllocVTable, Custom, ErrorBox, ErrorString},
    Error, ErrorKind,
};
use core::{alloc::Allocator, error, ptr, result};

unsafe fn set_alloc_vtable() {
    static ALLOC_VTABLE: AllocVTable =
        AllocVTable { deallocate: |ptr, layout| unsafe { Global.deallocate(ptr, layout) } };
    unsafe {
        ALLOC_VTABLE.install();
    }
}

fn into_error_box<T: ?Sized>(value: Box<T>) -> ErrorBox<T> {
    unsafe {
        set_alloc_vtable();
        ErrorBox::from_raw(Box::into_raw(value))
    }
}

fn into_box<T: ?Sized>(v: ErrorBox<T>) -> Box<T> {
    unsafe { Box::from_raw(v.into_raw()) }
}

impl From<String> for ErrorString {
    fn from(value: String) -> Self {
        unsafe {
            set_alloc_vtable();
            let (buf, length, capacity) = value.into_raw_parts();
            ErrorString::from_raw_parts(
                ErrorBox::from_raw(ptr::slice_from_raw_parts_mut(buf.cast(), capacity)).into(),
                length,
            )
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl From<crate::ffi::NulError> for Error {
    /// Converts a [`crate::ffi::NulError`] into a [`Error`].
    fn from(_: crate::ffi::NulError) -> Error {
        const_io_error!(ErrorKind::InvalidInput, "data provided contains a nul byte")
    }
}

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
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_allow_incoherent_impl]
    #[inline(never)]
    pub fn new<E>(kind: ErrorKind, error: E) -> Error
    where
        E: Into<Box<dyn error::Error + Send + Sync>>,
    {
        Self::_new(kind, error.into())
    }

    /// Creates a new I/O error from an arbitrary error payload.
    ///
    /// This function is used to generically create I/O errors which do not
    /// originate from the OS itself. It is a shortcut for [`Error::new`]
    /// with [`ErrorKind::Other`].
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
    ///
    /// [`Error::new`]: https://doc.rust-lang.org/std/io/struct.Error.html#method.new
    #[stable(feature = "io_error_other", since = "CURRENT_RUSTC_VERSION")]
    #[rustc_allow_incoherent_impl]
    pub fn other<E>(error: E) -> Error
    where
        E: Into<Box<dyn error::Error + Send + Sync>>,
    {
        Self::_new(ErrorKind::Other, error.into())
    }

    #[rustc_allow_incoherent_impl]
    fn _new(kind: ErrorKind, error: Box<dyn error::Error + Send + Sync>) -> Error {
        Error::_new_custom(into_error_box(Box::new(Custom { kind, error: into_error_box(error) })))
    }

    /// Consumes the `Error`, returning its inner error (if any).
    ///
    /// If this [`Error`] was constructed via [`new`] then this function will
    /// return [`Some`], otherwise it will return [`None`].
    ///
    /// [`new`]: https://doc.rust-lang.org/std/io/struct.Error.html#method.new
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
        self.into_inner_impl().map(into_box)
    }

    /// Attempt to downgrade the inner error to `E` if any.
    ///
    /// If this [`Error`] was constructed via [`new`] then this function will
    /// attempt to perform downgrade on it, otherwise it will return [`Err`].
    ///
    /// If downgrade succeeds, it will return [`Ok`], otherwise it will also
    /// return [`Err`].
    ///
    /// [`new`]: https://doc.rust-lang.org/std/io/struct.Error.html#method.new
    ///
    /// # Examples
    ///
    /// ```
    /// # #![allow(dead_code)]
    /// #![feature(io_error_downcast)]
    ///
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
    ///             .map(|b| *b)
    ///             .unwrap_or_else(E::Io)
    ///     }
    /// }
    /// ```
    #[unstable(feature = "io_error_downcast", issue = "99262")]
    #[rustc_allow_incoherent_impl]
    pub fn downcast<E>(self) -> result::Result<Box<E>, Self>
    where
        E: error::Error + Send + Sync + 'static,
    {
        self.downcast_impl::<E>().map(|p| unsafe { Box::from_raw(p) })
    }
}
