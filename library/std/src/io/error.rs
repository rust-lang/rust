#[cfg(test)]
mod tests;

#[stable(feature = "rust1", since = "1.0.0")]
pub use alloc_crate::io::ErrorKind;
#[unstable(feature = "raw_os_error_ty", issue = "107792")]
pub use alloc_crate::io::RawOsError;

// On 64-bit platforms, `io::Error` may use a bit-packed representation to
// reduce size. However, this representation assumes that error codes are
// always 32-bit wide.
//
// This assumption is invalid on 64-bit UEFI, where error codes are 64-bit.
// Therefore, the packed representation is explicitly disabled for UEFI
// targets, and the unpacked representation must be used instead.
#[cfg_attr(
    all(target_pointer_width = "64", not(target_os = "uefi")),
    path = "error/repr_bitpacked.rs"
)]
#[cfg_attr(
    not(all(target_pointer_width = "64", not(target_os = "uefi"))),
    path = "error/repr_unpacked.rs"
)]
mod repr;

#[cfg_attr(target_has_atomic_load_store = "ptr", path = "error/os_functions_atomic.rs")]
#[cfg_attr(not(target_has_atomic_load_store = "ptr"), path = "error/os_functions.rs")]
mod os_functions;

use self::os_functions::{decode_error_kind, format_os_error, is_interrupted, set_functions};
use self::repr::Repr;
use crate::{error, fmt, result};

/// A specialized [`Result`] type for I/O operations.
///
/// This type is broadly used across [`std::io`] for any operation which may
/// produce an error.
///
/// This type alias is generally used to avoid writing out [`io::Error`] directly and
/// is otherwise a direct mapping to [`Result`].
///
/// While usual Rust style is to import types directly, aliases of [`Result`]
/// often are not, to make it easier to distinguish between them. [`Result`] is
/// generally assumed to be [`core::result::Result`][`Result`], and so users of this alias
/// will generally use `io::Result` instead of shadowing the [prelude]'s import
/// of [`core::result::Result`][`Result`].
///
/// [`std::io`]: ../../std/io/index.html
/// [`io::Error`]: Error
/// [`Result`]: crate::result::Result
/// [prelude]: crate::prelude
///
/// # Examples
///
/// A convenience function that bubbles an `io::Result` to its caller:
///
/// ```
/// use std::io;
///
/// fn get_string() -> io::Result<String> {
///     let mut buffer = String::new();
///
///     io::stdin().read_line(&mut buffer)?;
///
///     Ok(buffer)
/// }
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
#[doc(search_unbox)]
pub type Result<T> = result::Result<T, Error>;

/// The error type for I/O operations of the [`Read`][Read], [`Write`][Write], [`Seek`][Seek], and
/// associated traits.
///
/// Errors mostly originate from the underlying OS, but custom instances of
/// `Error` can be created with crafted error messages and a particular value of
/// [`ErrorKind`].
///
/// [Read]: ../../std/io/trait.Read.html
/// [Write]: ../../std/io/trait.Write.html
/// [Seek]: ../../std/io/trait.Seek.html
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_has_incoherent_inherent_impls]
pub struct Error {
    repr: Repr,
}

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.repr, f)
    }
}

/// Common errors constants for use in std
#[doc(hidden)]
impl Error {
    #[doc(hidden)]
    #[unstable(feature = "core_io_internals", reason = "exposed only for libstd", issue = "none")]
    pub const INVALID_UTF8: Self =
        const_error!(ErrorKind::InvalidData, "stream did not contain valid UTF-8");

    #[doc(hidden)]
    #[unstable(feature = "core_io_internals", reason = "exposed only for libstd", issue = "none")]
    pub const READ_EXACT_EOF: Self =
        const_error!(ErrorKind::UnexpectedEof, "failed to fill whole buffer");

    #[doc(hidden)]
    #[unstable(feature = "core_io_internals", reason = "exposed only for libstd", issue = "none")]
    pub const UNKNOWN_THREAD_COUNT: Self = const_error!(
        ErrorKind::NotFound,
        "the number of hardware threads is not known for the target platform",
    );

    #[doc(hidden)]
    #[unstable(feature = "core_io_internals", reason = "exposed only for libstd", issue = "none")]
    pub const UNSUPPORTED_PLATFORM: Self =
        const_error!(ErrorKind::Unsupported, "operation not supported on this platform");

    #[doc(hidden)]
    #[unstable(feature = "core_io_internals", reason = "exposed only for libstd", issue = "none")]
    pub const WRITE_ALL_EOF: Self =
        const_error!(ErrorKind::WriteZero, "failed to write whole buffer");

    #[doc(hidden)]
    #[unstable(feature = "core_io_internals", reason = "exposed only for libstd", issue = "none")]
    pub const ZERO_TIMEOUT: Self =
        const_error!(ErrorKind::InvalidInput, "cannot set a 0 duration timeout");

    #[doc(hidden)]
    #[unstable(feature = "core_io_internals", reason = "exposed only for libstd", issue = "none")]
    pub const NO_ADDRESSES: Self =
        const_error!(ErrorKind::InvalidInput, "could not resolve to any addresses");
}

#[stable(feature = "rust1", since = "1.0.0")]
impl From<alloc::ffi::NulError> for Error {
    /// Converts a [`alloc::ffi::NulError`] into a [`Error`].
    fn from(_: alloc::ffi::NulError) -> Error {
        const_error!(ErrorKind::InvalidInput, "data provided contains a nul byte")
    }
}

#[stable(feature = "io_error_from_try_reserve", since = "1.78.0")]
impl From<alloc::collections::TryReserveError> for Error {
    /// Converts `TryReserveError` to an error with [`ErrorKind::OutOfMemory`].
    ///
    /// `TryReserveError` won't be available as the error `source()`,
    /// but this may change in the future.
    fn from(_: alloc::collections::TryReserveError) -> Error {
        // ErrorData::Custom allocates, which isn't great for handling OOM errors.
        ErrorKind::OutOfMemory.into()
    }
}

// Only derive debug in tests, to make sure it
// doesn't accidentally get printed.
#[cfg_attr(test, derive(Debug))]
enum ErrorData<C> {
    Os(RawOsError),
    Simple(ErrorKind),
    SimpleMessage(&'static SimpleMessage),
    Custom(C),
}

// `#[repr(align(4))]` is probably redundant, it should have that value or
// higher already. We include it just because repr_bitpacked.rs's encoding
// requires an alignment >= 4 (note that `#[repr(align)]` will not reduce the
// alignment required by the struct, only increase it).
//
// If we add more variants to ErrorData, this can be increased to 8, but it
// should probably be behind `#[cfg_attr(target_pointer_width = "64", ...)]` or
// whatever cfg we're using to enable the `repr_bitpacked` code, since only the
// that version needs the alignment, and 8 is higher than the alignment we'll
// have on 32 bit platforms.
//
// (For the sake of being explicit: the alignment requirement here only matters
// if `error/repr_bitpacked.rs` is in use — for the unpacked repr it doesn't
// matter at all)
#[doc(hidden)]
#[unstable(feature = "io_const_error_internals", issue = "none")]
#[repr(align(4))]
#[derive(Debug)]
pub struct SimpleMessage {
    pub kind: ErrorKind,
    pub message: &'static str,
}

/// Creates a new I/O error from a known kind of error and a string literal.
///
/// Contrary to [`Error::new`][new], this macro does not allocate and can be used in
/// `const` contexts.
///
/// [new]: ../../std/io/struct.Error.html#method.new
///
/// # Example
/// ```
/// #![feature(io_const_error)]
/// use std::io::{const_error, Error, ErrorKind};
///
/// const FAIL: Error = const_error!(ErrorKind::Unsupported, "tried something that never works");
///
/// fn not_here() -> Result<(), Error> {
///     Err(FAIL)
/// }
/// ```
#[rustc_macro_transparency = "semiopaque"]
#[unstable(feature = "io_const_error", issue = "133448")]
#[allow_internal_unstable(core_io, hint_must_use, io_const_error_internals)]
pub macro const_error($kind:expr, $message:expr $(,)?) {
    $crate::hint::must_use($crate::io::Error::from_static_message(
        const { &$crate::io::SimpleMessage { kind: $kind, message: $message } },
    ))
}

// As with `SimpleMessage`: `#[repr(align(4))]` here is just because
// repr_bitpacked's encoding requires it. In practice it almost certainly be
// already be this high or higher.
#[doc(hidden)]
#[repr(align(4))]
#[unstable(feature = "core_io_internals", reason = "exposed only for libstd", issue = "none")]
pub struct Custom {
    kind: ErrorKind,
    error: crate::ptr::NonNull<dyn error::Error + Send + Sync>,
    error_drop: unsafe fn(*mut (dyn error::Error + Send + Sync)),
    outer_drop: unsafe fn(*mut Self),
}

// SAFETY: All members of `Custom` are `Send`
unsafe impl Send for Custom {}

// SAFETY: All members of `Custom` are `Sync`
unsafe impl Sync for Custom {}

impl fmt::Debug for Custom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Custom").field("kind", &self.kind).field("error", self.error_ref()).finish()
    }
}

impl Drop for Custom {
    fn drop(&mut self) {
        // SAFETY: `Custom::from_raw` ensures this call is safe.
        unsafe {
            (self.error_drop)(self.error.as_ptr());
        }
    }
}

impl Custom {
    /// # Safety
    ///
    /// * `error` must be valid for up to a static lifetime, and own its pointee.
    /// * `error_drop` must be safe to call for the pointer `error` exactly once.
    /// * `outer_drop` must be safe to call on a pointer to this instance of `Custom`
    ///   if it were stored within a [`CustomOwner`].
    #[unstable(feature = "core_io_internals", reason = "exposed only for libstd", issue = "none")]
    pub unsafe fn from_raw(
        kind: ErrorKind,
        error: crate::ptr::NonNull<dyn error::Error + Send + Sync>,
        error_drop: unsafe fn(*mut (dyn error::Error + Send + Sync)),
        outer_drop: unsafe fn(*mut Self),
    ) -> Custom {
        Custom { kind, error, error_drop, outer_drop }
    }

    #[unstable(feature = "core_io_internals", reason = "exposed only for libstd", issue = "none")]
    pub fn into_raw(self) -> crate::ptr::NonNull<dyn error::Error + Send + Sync> {
        let ptr = self.error;
        core::mem::forget(self);
        ptr
    }

    fn error_ref(&self) -> &(dyn error::Error + Send + Sync + 'static) {
        // SAFETY:
        // `from_raw` ensures `error` is a valid pointer up to a static lifetime
        // and is owned by `self`
        unsafe { self.error.as_ref() }
    }

    fn error_mut(&mut self) -> &mut (dyn error::Error + Send + Sync + 'static) {
        // SAFETY:
        // `from_raw` ensures `error` is a valid pointer up to a static lifetime
        // and is owned by `self`
        unsafe { self.error.as_mut() }
    }
}

#[derive(Debug)]
#[repr(transparent)]
#[unstable(feature = "core_io_internals", reason = "exposed only for libstd", issue = "none")]
#[doc(hidden)]
pub struct CustomOwner(crate::ptr::NonNull<Custom>);

// SAFETY: Custom is `Send`
unsafe impl Send for CustomOwner {}

// SAFETY: Custom is `Sync`
unsafe impl Sync for CustomOwner {}

impl Drop for CustomOwner {
    fn drop(&mut self) {
        // SAFETY: `CustomOwner::from_raw` ensures this call is safe.
        unsafe {
            (self.0.as_ref().outer_drop)(self.0.as_ptr());
        }
    }
}

impl CustomOwner {
    /// # Safety
    ///
    /// * The `outer_drop` of the provided `custom` must be safe to call exactly once.
    #[unstable(feature = "core_io_internals", reason = "exposed only for libstd", issue = "none")]
    pub unsafe fn from_raw(custom: crate::ptr::NonNull<Custom>) -> CustomOwner {
        CustomOwner(custom)
    }

    #[unstable(feature = "core_io_internals", reason = "exposed only for libstd", issue = "none")]
    pub fn into_raw(self) -> crate::ptr::NonNull<Custom> {
        let ptr = self.0;
        core::mem::forget(self);
        ptr
    }

    #[allow(dead_code, reason = "only required for unpacked representation")]
    fn custom_ref(&self) -> &Custom {
        // SAFETY:
        // `from_raw` ensures `0` is a valid pointer up to a static lifetime
        // and is owned by `self`
        unsafe { self.0.as_ref() }
    }

    #[allow(dead_code, reason = "only required for unpacked representation")]
    fn custom_mut(&mut self) -> &mut Custom {
        // SAFETY:
        // `from_raw` ensures `0` is a valid pointer up to a static lifetime
        // and is owned by `self`
        unsafe { self.0.as_mut() }
    }
}

/// Intended for use for errors not exposed to the user, where allocating onto
/// the heap (for normal construction via Error::new) is too costly.
#[stable(feature = "io_error_from_errorkind", since = "1.14.0")]
impl From<ErrorKind> for Error {
    /// Converts an [`ErrorKind`] into an [`Error`].
    ///
    /// This conversion creates a new error with a simple representation of error kind.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::io::{Error, ErrorKind};
    ///
    /// let not_found = ErrorKind::NotFound;
    /// let error = Error::from(not_found);
    /// assert_eq!("entity not found", format!("{error}"));
    /// ```
    #[inline]
    fn from(kind: ErrorKind) -> Error {
        Error { repr: Repr::new_simple(kind) }
    }
}

impl Error {
    /// # Safety
    ///
    /// The provided `CustomOwner` must have been constructed from a `Box` from the `alloc` crate.
    #[doc(hidden)]
    #[unstable(feature = "core_io_internals", reason = "exposed only for libstd", issue = "none")]
    #[must_use]
    #[inline]
    pub unsafe fn from_custom_owner(custom: CustomOwner) -> Error {
        Error { repr: Repr::new_custom(custom) }
    }

    #[doc(hidden)]
    #[unstable(feature = "core_io_internals", reason = "exposed only for libstd", issue = "none")]
    #[must_use]
    #[inline]
    pub fn into_custom_owner(self) -> result::Result<CustomOwner, Self> {
        if matches!(self.repr.data(), ErrorData::Custom(..)) {
            let ErrorData::Custom(c) = self.repr.into_data() else {
                // SAFETY: Checked above using `matches!`.
                unsafe { crate::hint::unreachable_unchecked() }
            };
            Ok(c)
        } else {
            Err(self)
        }
    }

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
    #[stable(feature = "io_error_other", since = "1.74.0")]
    #[rustc_allow_incoherent_impl]
    pub fn other<E>(error: E) -> Error
    where
        E: Into<Box<dyn error::Error + Send + Sync>>,
    {
        error_from_box(ErrorKind::Other, error.into())
    }

    /// Creates a new I/O error from a known kind of error as well as a constant
    /// message.
    ///
    /// This function does not allocate.
    ///
    /// You should not use this directly, and instead use the `const_error!`
    /// macro: `io::const_error!(ErrorKind::Something, "some_message")`.
    ///
    /// This function should maybe change to `from_static_message<const MSG: &'static
    /// str>(kind: ErrorKind)` in the future, when const generics allow that.
    #[inline]
    #[doc(hidden)]
    #[unstable(feature = "io_const_error_internals", issue = "none")]
    pub const fn from_static_message(msg: &'static SimpleMessage) -> Error {
        Self { repr: Repr::new_simple_message(msg) }
    }

    /// # Safety
    ///
    /// `functions` must point to data that is entirely constant; it must
    /// not be created during runtime.
    #[doc(hidden)]
    #[unstable(feature = "core_io_internals", reason = "exposed only for libstd", issue = "none")]
    #[must_use]
    #[inline]
    pub unsafe fn from_raw_os_error_with_functions(
        code: RawOsError,
        functions: &'static OsFunctions,
    ) -> Error {
        // SAFETY: Caller ensures `functions` is a constant not created at runtime.
        unsafe {
            set_functions(functions);
        }
        Error { repr: Repr::new_os(code) }
    }

    /// Returns an error representing the last OS error which occurred.
    ///
    /// This function reads the value of `errno` for the target platform (e.g.
    /// `GetLastError` on Windows) and will return a corresponding instance of
    /// [`Error`] for the error code.
    ///
    /// This should be called immediately after a call to a platform function,
    /// otherwise the state of the error value is indeterminate. In particular,
    /// other standard library functions may call platform functions that may
    /// (or may not) reset the error value even if they succeed.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::io::Error;
    ///
    /// let os_error = Error::last_os_error();
    /// println!("last OS error: {os_error:?}");
    /// ```
    #[rustc_allow_incoherent_impl]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[doc(alias = "GetLastError")]
    #[doc(alias = "errno")]
    #[must_use]
    #[inline]
    pub fn last_os_error() -> Error {
        Error::from_raw_os_error(errno())
    }

    /// Creates a new instance of an [`Error`] from a particular OS error code.
    ///
    /// # Examples
    ///
    /// On Linux:
    ///
    /// ```
    /// # if cfg!(target_os = "linux") {
    /// use std::io;
    ///
    /// let error = io::Error::from_raw_os_error(22);
    /// assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
    /// # }
    /// ```
    ///
    /// On Windows:
    ///
    /// ```
    /// # if cfg!(windows) {
    /// use std::io;
    ///
    /// let error = io::Error::from_raw_os_error(10022);
    /// assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
    /// # }
    /// ```
    #[rustc_allow_incoherent_impl]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[must_use]
    #[inline]
    pub fn from_raw_os_error(code: RawOsError) -> Error {
        const FUNCTIONS: &'static OsFunctions = &OsFunctions {
            format_os_error: |code, fmt| fmt.write_str(&error_string(code)),
            decode_error_kind,
            is_interrupted,
        };

        // SAFETY: `FUNCTIONS` is a constant and not created at runtime.
        unsafe { Error::from_raw_os_error_with_functions(code, FUNCTIONS) }
    }

    /// Returns the OS error that this error represents (if any).
    ///
    /// If this [`Error`] was constructed via [`last_os_error`][last_os_error] or
    /// [`from_raw_os_error`][from_raw_os_error], then this function will return [`Some`], otherwise
    /// it will return [`None`].
    ///
    /// [last_os_error]: ../../std/io/struct.Error.html#method.last_os_error
    /// [from_raw_os_error]: ../../std/io/struct.Error.html#method.from_raw_os_error
    ///
    /// # Examples
    ///
    /// ```
    /// use std::io::{Error, ErrorKind};
    ///
    /// fn print_os_error(err: &Error) {
    ///     if let Some(raw_os_err) = err.raw_os_error() {
    ///         println!("raw OS error: {raw_os_err:?}");
    ///     } else {
    ///         println!("Not an OS error");
    ///     }
    /// }
    ///
    /// fn main() {
    ///     // Will print "raw OS error: ...".
    ///     print_os_error(&Error::last_os_error());
    ///     // Will print "Not an OS error".
    ///     print_os_error(&Error::new(ErrorKind::Other, "oh no!"));
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[must_use]
    #[inline]
    pub fn raw_os_error(&self) -> Option<RawOsError> {
        match self.repr.data() {
            ErrorData::Os(i) => Some(i),
            ErrorData::Custom(..) => None,
            ErrorData::Simple(..) => None,
            ErrorData::SimpleMessage(..) => None,
        }
    }

    /// Returns a reference to the inner error wrapped by this error (if any).
    ///
    /// If this [`Error`] was constructed via [`new`][new] then this function will
    /// return [`Some`], otherwise it will return [`None`].
    ///
    /// [new]: ../../std/io/struct.Error.html#method.new
    ///
    /// # Examples
    ///
    /// ```
    /// use std::io::{Error, ErrorKind};
    ///
    /// fn print_error(err: &Error) {
    ///     if let Some(inner_err) = err.get_ref() {
    ///         println!("Inner error: {inner_err:?}");
    ///     } else {
    ///         println!("No inner error");
    ///     }
    /// }
    ///
    /// fn main() {
    ///     // Will print "No inner error".
    ///     print_error(&Error::last_os_error());
    ///     // Will print "Inner error: ...".
    ///     print_error(&Error::new(ErrorKind::Other, "oh no!"));
    /// }
    /// ```
    #[stable(feature = "io_error_inner", since = "1.3.0")]
    #[must_use]
    #[inline]
    pub fn get_ref(&self) -> Option<&(dyn error::Error + Send + Sync + 'static)> {
        match self.repr.data() {
            ErrorData::Os(..) => None,
            ErrorData::Simple(..) => None,
            ErrorData::SimpleMessage(..) => None,
            ErrorData::Custom(c) => Some(c.error_ref()),
        }
    }

    /// Returns a mutable reference to the inner error wrapped by this error
    /// (if any).
    ///
    /// If this [`Error`] was constructed via [`new`][new] then this function will
    /// return [`Some`], otherwise it will return [`None`].
    ///
    /// [new]: ../../std/io/struct.Error.html#method.new
    ///
    /// # Examples
    ///
    /// ```
    /// use std::io::{Error, ErrorKind};
    /// use std::{error, fmt};
    /// use std::fmt::Display;
    ///
    /// #[derive(Debug)]
    /// struct MyError {
    ///     v: String,
    /// }
    ///
    /// impl MyError {
    ///     fn new() -> MyError {
    ///         MyError {
    ///             v: "oh no!".to_string()
    ///         }
    ///     }
    ///
    ///     fn change_message(&mut self, new_message: &str) {
    ///         self.v = new_message.to_string();
    ///     }
    /// }
    ///
    /// impl error::Error for MyError {}
    ///
    /// impl Display for MyError {
    ///     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    ///         write!(f, "MyError: {}", self.v)
    ///     }
    /// }
    ///
    /// fn change_error(mut err: Error) -> Error {
    ///     if let Some(inner_err) = err.get_mut() {
    ///         inner_err.downcast_mut::<MyError>().unwrap().change_message("I've been changed!");
    ///     }
    ///     err
    /// }
    ///
    /// fn print_error(err: &Error) {
    ///     if let Some(inner_err) = err.get_ref() {
    ///         println!("Inner error: {inner_err}");
    ///     } else {
    ///         println!("No inner error");
    ///     }
    /// }
    ///
    /// fn main() {
    ///     // Will print "No inner error".
    ///     print_error(&change_error(Error::last_os_error()));
    ///     // Will print "Inner error: ...".
    ///     print_error(&change_error(Error::new(ErrorKind::Other, MyError::new())));
    /// }
    /// ```
    #[stable(feature = "io_error_inner", since = "1.3.0")]
    #[must_use]
    #[inline]
    pub fn get_mut(&mut self) -> Option<&mut (dyn error::Error + Send + Sync + 'static)> {
        match self.repr.data_mut() {
            ErrorData::Os(..) => None,
            ErrorData::Simple(..) => None,
            ErrorData::SimpleMessage(..) => None,
            ErrorData::Custom(c) => Some(c.error_mut()),
        }
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

    /// Returns the corresponding [`ErrorKind`] for this error.
    ///
    /// This may be a value set by Rust code constructing custom `io::Error`s,
    /// or if this `io::Error` was sourced from the operating system,
    /// it will be a value inferred from the system's error encoding.
    /// See [`last_os_error`][last_os_error] for more details.
    ///
    /// [last_os_error]: ../../std/io/struct.Error.html#method.last_os_error
    ///
    /// # Examples
    ///
    /// ```
    /// use std::io::{Error, ErrorKind};
    ///
    /// fn print_error(err: Error) {
    ///     println!("{:?}", err.kind());
    /// }
    ///
    /// fn main() {
    ///     // As no error has (visibly) occurred, this may print anything!
    ///     // It likely prints a placeholder for unidentified (non-)errors.
    ///     print_error(Error::last_os_error());
    ///     // Will print "AddrInUse".
    ///     print_error(Error::new(ErrorKind::AddrInUse, "oh no!"));
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[must_use]
    #[inline]
    pub fn kind(&self) -> ErrorKind {
        match self.repr.data() {
            ErrorData::Os(code) => decode_error_kind(code),
            ErrorData::Custom(c) => c.kind,
            ErrorData::Simple(kind) => kind,
            ErrorData::SimpleMessage(m) => m.kind,
        }
    }

    #[unstable(feature = "core_io_internals", reason = "exposed only for libstd", issue = "none")]
    #[doc(hidden)]
    #[inline]
    pub fn is_interrupted(&self) -> bool {
        match self.repr.data() {
            ErrorData::Os(code) => is_interrupted(code),
            ErrorData::Custom(c) => c.kind == ErrorKind::Interrupted,
            ErrorData::Simple(kind) => kind == ErrorKind::Interrupted,
            ErrorData::SimpleMessage(m) => m.kind == ErrorKind::Interrupted,
        }
    }
}

impl fmt::Debug for Repr {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.data() {
            ErrorData::Os(code) => fmt
                .debug_struct("Os")
                .field("code", &code)
                .field("kind", &decode_error_kind(code))
                .field(
                    "message",
                    &fmt::from_fn(|fmt| {
                        write!(fmt, "\"{}\"", fmt::from_fn(|fmt| format_os_error(code, fmt)))
                    }),
                )
                .finish(),
            ErrorData::Custom(c) => fmt::Debug::fmt(&c, fmt),
            ErrorData::Simple(kind) => fmt.debug_tuple("Kind").field(&kind).finish(),
            ErrorData::SimpleMessage(msg) => fmt
                .debug_struct("Error")
                .field("kind", &msg.kind)
                .field("message", &msg.message)
                .finish(),
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Display for Error {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.repr.data() {
            ErrorData::Os(code) => {
                let detail = fmt::from_fn(|fmt| format_os_error(code, fmt));
                write!(fmt, "{detail} (os error {code})")
            }
            ErrorData::Custom(c) => fmt::Display::fmt(c.error_ref(), fmt),
            ErrorData::Simple(kind) => kind.fmt(fmt),
            ErrorData::SimpleMessage(msg) => msg.message.fmt(fmt),
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl error::Error for Error {
    #[allow(deprecated)]
    fn cause(&self) -> Option<&dyn error::Error> {
        match self.repr.data() {
            ErrorData::Os(..) => None,
            ErrorData::Simple(..) => None,
            ErrorData::SimpleMessage(..) => None,
            ErrorData::Custom(c) => c.error_ref().cause(),
        }
    }

    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self.repr.data() {
            ErrorData::Os(..) => None,
            ErrorData::Simple(..) => None,
            ErrorData::SimpleMessage(..) => None,
            ErrorData::Custom(c) => c.error_ref().source(),
        }
    }
}

fn _assert_error_is_sync_send() {
    fn _is_sync_send<T: Sync + Send>() {}
    _is_sync_send::<Error>();
}

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
