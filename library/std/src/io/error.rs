#[cfg(test)]
mod tests;

use crate::convert::From;
use crate::error;
use crate::fmt;
use crate::result;
use crate::sys;

/// A specialized [`Result`] type for I/O operations.
///
/// This type is broadly used across [`std::io`] for any operation which may
/// produce an error.
///
/// This typedef is generally used to avoid writing out [`io::Error`] directly and
/// is otherwise a direct mapping to [`Result`].
///
/// While usual Rust style is to import types directly, aliases of [`Result`]
/// often are not, to make it easier to distinguish between them. [`Result`] is
/// generally assumed to be [`std::result::Result`][`Result`], and so users of this alias
/// will generally use `io::Result` instead of shadowing the [prelude]'s import
/// of [`std::result::Result`][`Result`].
///
/// [`std::io`]: crate::io
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
pub type Result<T> = result::Result<T, Error>;

/// The error type for I/O operations of the [`Read`], [`Write`], [`Seek`], and
/// associated traits.
///
/// Errors mostly originate from the underlying OS, but custom instances of
/// `Error` can be created with crafted error messages and a particular value of
/// [`ErrorKind`].
///
/// [`Read`]: crate::io::Read
/// [`Write`]: crate::io::Write
/// [`Seek`]: crate::io::Seek
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Error {
    repr: Box<dyn IoError>,
}

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.repr, f)
    }
}
trait AsError: error::Error + Send + Sync {
    fn as_error_ref(&self) -> &(dyn error::Error + Send + Sync + 'static);
    fn as_error_mut(&mut self) -> &mut (dyn error::Error + Send + Sync + 'static);
    fn into_error(self: Box<Self>) -> Box<(dyn error::Error + Send + Sync + 'static)>;
}
impl<T: error::Error + Send + Sync + 'static> AsError for T {
    fn as_error_ref(&self) -> &(dyn error::Error + Send + Sync + 'static) {
        self
    }
    fn as_error_mut(&mut self) -> &mut (dyn error::Error + Send + Sync + 'static) {
        self
    }
    fn into_error(self: Box<Self>) -> Box<(dyn error::Error + Send + Sync + 'static)> {
        self
    }
}
trait IoError: AsError {
    fn kind(&self) -> ErrorKind;
}

struct Os;

impl Os {
    fn code(&self) -> i32 {
        let code = unsafe { core::mem::transmute::<_, isize>(self) } as i32;
        if code == i32::MIN { 0 } else { code }
    }
}

impl fmt::Debug for Os {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.debug_struct("Os")
            .field("code", &self.code())
            .field("kind", &self.kind())
            .field("message", &sys::os::error_string(self.code()))
            .finish()
    }
}

impl fmt::Display for Os {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        let detail = sys::os::error_string(self.code());
        write!(fmt, "{} (os error {})", detail, self.code())
    }
}

impl error::Error for Os {
    #[allow(deprecated, deprecated_in_future)]
    fn description(&self) -> &str {
        self.kind().as_str()
    }
}

impl IoError for Os {
    fn kind(&self) -> ErrorKind {
        sys::decode_error_kind(self.code())
    }
}

#[derive(Debug)]
struct Custom {
    kind: ErrorKind,
    error: Box<dyn error::Error + Send + Sync>,
}

impl IoError for Custom {
    fn kind(&self) -> ErrorKind {
        self.kind
    }
}

impl fmt::Display for Custom {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.error.fmt(fmt)
    }
}

impl error::Error for Custom {
    #[allow(deprecated, deprecated_in_future)]
    fn description(&self) -> &str {
        self.error.description()
    }

    #[allow(deprecated)]
    fn cause(&self) -> Option<&dyn error::Error> {
        self.error.cause()
    }

    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        self.error.source()
    }
}

/// A list specifying general categories of I/O error.
///
/// This list is intended to grow over time and it is not recommended to
/// exhaustively match against it.
///
/// It is used with the [`io::Error`] type.
///
/// [`io::Error`]: Error
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[stable(feature = "rust1", since = "1.0.0")]
#[allow(deprecated)]
#[non_exhaustive]
pub enum ErrorKind {
    /// An entity was not found, often a file.
    #[stable(feature = "rust1", since = "1.0.0")]
    NotFound,
    /// The operation lacked the necessary privileges to complete.
    #[stable(feature = "rust1", since = "1.0.0")]
    PermissionDenied,
    /// The connection was refused by the remote server.
    #[stable(feature = "rust1", since = "1.0.0")]
    ConnectionRefused,
    /// The connection was reset by the remote server.
    #[stable(feature = "rust1", since = "1.0.0")]
    ConnectionReset,
    /// The connection was aborted (terminated) by the remote server.
    #[stable(feature = "rust1", since = "1.0.0")]
    ConnectionAborted,
    /// The network operation failed because it was not connected yet.
    #[stable(feature = "rust1", since = "1.0.0")]
    NotConnected,
    /// A socket address could not be bound because the address is already in
    /// use elsewhere.
    #[stable(feature = "rust1", since = "1.0.0")]
    AddrInUse,
    /// A nonexistent interface was requested or the requested address was not
    /// local.
    #[stable(feature = "rust1", since = "1.0.0")]
    AddrNotAvailable,
    /// The operation failed because a pipe was closed.
    #[stable(feature = "rust1", since = "1.0.0")]
    BrokenPipe,
    /// An entity already exists, often a file.
    #[stable(feature = "rust1", since = "1.0.0")]
    AlreadyExists,
    /// The operation needs to block to complete, but the blocking operation was
    /// requested to not occur.
    #[stable(feature = "rust1", since = "1.0.0")]
    WouldBlock,
    /// A parameter was incorrect.
    #[stable(feature = "rust1", since = "1.0.0")]
    InvalidInput,
    /// Data not valid for the operation were encountered.
    ///
    /// Unlike [`InvalidInput`], this typically means that the operation
    /// parameters were valid, however the error was caused by malformed
    /// input data.
    ///
    /// For example, a function that reads a file into a string will error with
    /// `InvalidData` if the file's contents are not valid UTF-8.
    ///
    /// [`InvalidInput`]: ErrorKind::InvalidInput
    #[stable(feature = "io_invalid_data", since = "1.2.0")]
    InvalidData,
    /// The I/O operation's timeout expired, causing it to be canceled.
    #[stable(feature = "rust1", since = "1.0.0")]
    TimedOut,
    /// An error returned when an operation could not be completed because a
    /// call to [`write`] returned [`Ok(0)`].
    ///
    /// This typically means that an operation could only succeed if it wrote a
    /// particular number of bytes but only a smaller number of bytes could be
    /// written.
    ///
    /// [`write`]: crate::io::Write::write
    /// [`Ok(0)`]: Ok
    #[stable(feature = "rust1", since = "1.0.0")]
    WriteZero,
    /// This operation was interrupted.
    ///
    /// Interrupted operations can typically be retried.
    #[stable(feature = "rust1", since = "1.0.0")]
    Interrupted,
    /// Any I/O error not part of this list.
    ///
    /// Errors that are `Other` now may move to a different or a new
    /// [`ErrorKind`] variant in the future. It is not recommended to match
    /// an error against `Other` and to expect any additional characteristics,
    /// e.g., a specific [`Error::raw_os_error`] return value.
    #[stable(feature = "rust1", since = "1.0.0")]
    Other,

    /// An error returned when an operation could not be completed because an
    /// "end of file" was reached prematurely.
    ///
    /// This typically means that an operation could only succeed if it read a
    /// particular number of bytes but only a smaller number of bytes could be
    /// read.
    #[stable(feature = "read_exact", since = "1.6.0")]
    UnexpectedEof,
}

impl ErrorKind {
    pub(crate) fn as_str(&self) -> &'static str {
        match *self {
            ErrorKind::NotFound => "entity not found",
            ErrorKind::PermissionDenied => "permission denied",
            ErrorKind::ConnectionRefused => "connection refused",
            ErrorKind::ConnectionReset => "connection reset",
            ErrorKind::ConnectionAborted => "connection aborted",
            ErrorKind::NotConnected => "not connected",
            ErrorKind::AddrInUse => "address in use",
            ErrorKind::AddrNotAvailable => "address not available",
            ErrorKind::BrokenPipe => "broken pipe",
            ErrorKind::AlreadyExists => "entity already exists",
            ErrorKind::WouldBlock => "operation would block",
            ErrorKind::InvalidInput => "invalid input parameter",
            ErrorKind::InvalidData => "invalid data",
            ErrorKind::TimedOut => "timed out",
            ErrorKind::WriteZero => "write zero",
            ErrorKind::Interrupted => "operation interrupted",
            ErrorKind::Other => "other os error",
            ErrorKind::UnexpectedEof => "unexpected end of file",
        }
    }
}

struct ErrorKindZst;

impl fmt::Debug for ErrorKindZst {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.debug_tuple("Kind").field(&self.kind()).finish()
    }
}

impl fmt::Display for ErrorKindZst {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(fmt, "{}", self.kind().as_str())
    }
}

impl error::Error for ErrorKindZst {
    #[allow(deprecated, deprecated_in_future)]
    fn description(&self) -> &str {
        self.kind().as_str()
    }
}

impl IoError for ErrorKindZst {
    fn kind(&self) -> ErrorKind {
        u8_to_error_kind(self as *const ErrorKindZst as usize as u8)
    }
}
macro_rules! error_kind_to_u8 {
    ($kind:expr => $m:ident) => {
        match $kind {
            ErrorKind::NotFound => $m!(1),
            ErrorKind::PermissionDenied => $m!(2),
            ErrorKind::ConnectionRefused => $m!(3),
            ErrorKind::ConnectionReset => $m!(4),
            ErrorKind::ConnectionAborted => $m!(5),
            ErrorKind::NotConnected => $m!(6),
            ErrorKind::AddrInUse => $m!(7),
            ErrorKind::AddrNotAvailable => $m!(8),
            ErrorKind::BrokenPipe => $m!(9),
            ErrorKind::AlreadyExists => $m!(10),
            ErrorKind::WouldBlock => $m!(11),
            ErrorKind::InvalidInput => $m!(12),
            ErrorKind::InvalidData => $m!(13),
            ErrorKind::TimedOut => $m!(14),
            ErrorKind::WriteZero => $m!(15),
            ErrorKind::Interrupted => $m!(16),
            ErrorKind::Other => $m!(17),
            ErrorKind::UnexpectedEof => $m!(18),
        }
    };
}
#[inline(always)]
fn u8_to_error_kind(n: u8) -> ErrorKind {
    match n {
        1 => ErrorKind::NotFound,
        2 => ErrorKind::PermissionDenied,
        3 => ErrorKind::ConnectionRefused,
        4 => ErrorKind::ConnectionReset,
        5 => ErrorKind::ConnectionAborted,
        6 => ErrorKind::NotConnected,
        7 => ErrorKind::AddrInUse,
        8 => ErrorKind::AddrNotAvailable,
        9 => ErrorKind::BrokenPipe,
        10 => ErrorKind::AlreadyExists,
        11 => ErrorKind::WouldBlock,
        12 => ErrorKind::InvalidInput,
        13 => ErrorKind::InvalidData,
        14 => ErrorKind::TimedOut,
        15 => ErrorKind::WriteZero,
        16 => ErrorKind::Interrupted,
        17 => ErrorKind::Other,
        18 => ErrorKind::UnexpectedEof,
        _ => unreachable!(),
    }
}
/// Intended for use for errors not exposed to the user, where allocating onto
/// the heap (for normal construction via Error::new) is too costly.
#[stable(feature = "io_error_from_errorkind", since = "1.14.0")]
impl From<ErrorKind> for Error {
    /// Converts an [`ErrorKind`] into an [`Error`].
    ///
    /// This conversion allocates a new error with a simple representation of error kind.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::io::{Error, ErrorKind};
    ///
    /// let not_found = ErrorKind::NotFound;
    /// let error = Error::from(not_found);
    /// assert_eq!("entity not found", format!("{}", error));
    /// ```
    #[inline]
    fn from(kind: ErrorKind) -> Error {
        macro_rules! as_usize {
            ($n:literal) => {
                $n as usize
            };
        }
        let n: usize = error_kind_to_u8!(kind => as_usize);
        Error { repr: unsafe { Box::from_raw(n as *mut ErrorKindZst) } }
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
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn new<E>(kind: ErrorKind, error: E) -> Error
    where
        E: Into<Box<dyn error::Error + Send + Sync>>,
    {
        Self::_new(kind, error.into())
    }

    fn _new(kind: ErrorKind, error: Box<dyn error::Error + Send + Sync>) -> Error {
        Error { repr: Box::new(Custom { kind, error }) }
    }

    /// Creates a new I/O error from a known kind of error as well as a
    /// constant message.
    ///
    /// This function does not allocate.
    ///
    /// This function should maybe change to
    /// `new_const<const MSG: &'static str>(kind: ErrorKind)`
    /// in the future, when const generics allow that.
    #[inline]
    #[rustc_allow_const_fn_unstable(const_fn, const_box_from_raw)]
    pub(crate) const fn new_const(kind: ErrorKind, message: &'static &'static str) -> Error {
        struct SimpleMessage<const KIND: u8>;

        impl<const KIND: u8> SimpleMessage<KIND> {
            fn as_str(&self) -> &str {
                unsafe { *(self as *const _ as *const &'static str) }
            }
        }
        impl<const KIND: u8> core::fmt::Debug for SimpleMessage<KIND> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_struct("Custom")
                    .field("kind", &self.kind())
                    .field("error", &self.as_str())
                    .finish()
            }
        }
        impl<const KIND: u8> core::fmt::Display for SimpleMessage<KIND> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(self.as_str())
            }
        }
        impl<const KIND: u8> error::Error for SimpleMessage<KIND> {
            #[allow(deprecated, deprecated_in_future)]
            fn description(&self) -> &str {
                self.as_str()
            }
        }
        impl<const KIND: u8> IoError for SimpleMessage<KIND> {
            fn kind(&self) -> ErrorKind {
                u8_to_error_kind(KIND)
            }
        }
        macro_rules! message {
            ($n:literal) => {
                Box::<SimpleMessage<{ $n }>>::from_raw_in(
                    message as *const &'static str as *mut &'static str as *mut _,
                    crate::alloc::Global,
                )
            };
        }
        Self { repr: unsafe { error_kind_to_u8!(kind => message) } }
    }

    /// Returns an error representing the last OS error which occurred.
    ///
    /// This function reads the value of `errno` for the target platform (e.g.
    /// `GetLastError` on Windows) and will return a corresponding instance of
    /// [`Error`] for the error code.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::io::Error;
    ///
    /// println!("last OS error: {:?}", Error::last_os_error());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn last_os_error() -> Error {
        Error::from_raw_os_error(sys::os::errno() as i32)
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
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn from_raw_os_error(code: i32) -> Error {
        assert_ne!(code, i32::MIN, "`i32::MIN` is not a valid error code");
        Error {
            repr: unsafe {
                Box::from_raw(core::mem::transmute::<_, *mut Os>(if code == 0 {
                    i32::MIN
                } else {
                    code
                } as isize))
            },
        }
    }

    /// Returns the OS error that this error represents (if any).
    ///
    /// If this [`Error`] was constructed via [`last_os_error`] or
    /// [`from_raw_os_error`], then this function will return [`Some`], otherwise
    /// it will return [`None`].
    ///
    /// [`last_os_error`]: Error::last_os_error
    /// [`from_raw_os_error`]: Error::from_raw_os_error
    ///
    /// # Examples
    ///
    /// ```
    /// use std::io::{Error, ErrorKind};
    ///
    /// fn print_os_error(err: &Error) {
    ///     if let Some(raw_os_err) = err.raw_os_error() {
    ///         println!("raw OS error: {:?}", raw_os_err);
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
    #[inline]
    pub fn raw_os_error(&self) -> Option<i32> {
        self.repr.as_error_ref().downcast_ref::<Os>().map(|os| os.code())
    }

    /// Returns a reference to the inner error wrapped by this error (if any).
    ///
    /// If this [`Error`] was constructed via [`new`] then this function will
    /// return [`Some`], otherwise it will return [`None`].
    ///
    /// [`new`]: Error::new
    ///
    /// # Examples
    ///
    /// ```
    /// use std::io::{Error, ErrorKind};
    ///
    /// fn print_error(err: &Error) {
    ///     if let Some(inner_err) = err.get_ref() {
    ///         println!("Inner error: {:?}", inner_err);
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
    #[inline]
    pub fn get_ref(&self) -> Option<&(dyn error::Error + Send + Sync + 'static)> {
        self.repr.as_error_ref().downcast_ref::<Custom>().map(|c| &*c.error)
    }

    /// Returns a mutable reference to the inner error wrapped by this error
    /// (if any).
    ///
    /// If this [`Error`] was constructed via [`new`] then this function will
    /// return [`Some`], otherwise it will return [`None`].
    ///
    /// [`new`]: Error::new
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
    ///         write!(f, "MyError: {}", &self.v)
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
    ///         println!("Inner error: {}", inner_err);
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
    #[inline]
    pub fn get_mut(&mut self) -> Option<&mut (dyn error::Error + Send + Sync + 'static)> {
        self.repr.as_error_mut().downcast_mut::<Custom>().map(|c| &mut *c.error)
    }

    /// Consumes the `Error`, returning its inner error (if any).
    ///
    /// If this [`Error`] was constructed via [`new`] then this function will
    /// return [`Some`], otherwise it will return [`None`].
    ///
    /// [`new`]: Error::new
    ///
    /// # Examples
    ///
    /// ```
    /// use std::io::{Error, ErrorKind};
    ///
    /// fn print_error(err: Error) {
    ///     if let Some(inner_err) = err.into_inner() {
    ///         println!("Inner error: {}", inner_err);
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
    #[inline]
    pub fn into_inner(self) -> Option<Box<dyn error::Error + Send + Sync>> {
        self.repr.into_error().downcast::<Custom>().ok().map(|c| c.error)
    }

    /// Returns the corresponding [`ErrorKind`] for this error.
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
    ///     // Will print "Other".
    ///     print_error(Error::last_os_error());
    ///     // Will print "AddrInUse".
    ///     print_error(Error::new(ErrorKind::AddrInUse, "oh no!"));
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn kind(&self) -> ErrorKind {
        self.repr.kind()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Display for Error {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.repr.fmt(fmt)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl error::Error for Error {
    #[allow(deprecated, deprecated_in_future)]
    fn description(&self) -> &str {
        self.repr.description()
    }

    #[allow(deprecated)]
    fn cause(&self) -> Option<&dyn error::Error> {
        self.repr.cause()
    }

    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        self.repr.source()
    }
}

fn _assert_error_is_sync_send() {
    fn _is_sync_send<T: Sync + Send>() {}
    _is_sync_send::<Error>();
}
