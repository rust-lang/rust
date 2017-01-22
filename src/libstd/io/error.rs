// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use error;
use fmt;
use result;
use sys;
use convert::From;

/// A specialized [`Result`](../result/enum.Result.html) type for I/O
/// operations.
///
/// This type is broadly used across `std::io` for any operation which may
/// produce an error.
///
/// This typedef is generally used to avoid writing out `io::Error` directly and
/// is otherwise a direct mapping to `Result`.
///
/// While usual Rust style is to import types directly, aliases of `Result`
/// often are not, to make it easier to distinguish between them. `Result` is
/// generally assumed to be `std::result::Result`, and so users of this alias
/// will generally use `io::Result` instead of shadowing the prelude's import
/// of `std::result::Result`.
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

/// The error type for I/O operations of the `Read`, `Write`, `Seek`, and
/// associated traits.
///
/// Errors mostly originate from the underlying OS, but custom instances of
/// `Error` can be created with crafted error messages and a particular value of
/// [`ErrorKind`].
///
/// [`ErrorKind`]: enum.ErrorKind.html
#[derive(Debug)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Error {
    repr: Repr,
}

enum Repr {
    Os(i32),
    Simple(ErrorKind),
    Custom(Box<Custom>),
}

#[derive(Debug)]
struct Custom {
    kind: ErrorKind,
    error: Box<error::Error+Send+Sync>,
}

/// A list specifying general categories of I/O error.
///
/// This list is intended to grow over time and it is not recommended to
/// exhaustively match against it.
///
/// It is used with the [`io::Error`] type.
///
/// [`io::Error`]: struct.Error.html
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[stable(feature = "rust1", since = "1.0.0")]
#[allow(deprecated)]
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
    /// [`InvalidInput`]: #variant.InvalidInput
    #[stable(feature = "io_invalid_data", since = "1.2.0")]
    InvalidData,
    /// The I/O operation's timeout expired, causing it to be canceled.
    #[stable(feature = "rust1", since = "1.0.0")]
    TimedOut,
    /// An error returned when an operation could not be completed because a
    /// call to [`write()`] returned [`Ok(0)`].
    ///
    /// This typically means that an operation could only succeed if it wrote a
    /// particular number of bytes but only a smaller number of bytes could be
    /// written.
    ///
    /// [`write()`]: ../../std/io/trait.Write.html#tymethod.write
    /// [`Ok(0)`]: ../../std/io/type.Result.html
    #[stable(feature = "rust1", since = "1.0.0")]
    WriteZero,
    /// This operation was interrupted.
    ///
    /// Interrupted operations can typically be retried.
    #[stable(feature = "rust1", since = "1.0.0")]
    Interrupted,
    /// Any I/O error not part of this list.
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

    /// A marker variant that tells the compiler that users of this enum cannot
    /// match it exhaustively.
    #[unstable(feature = "io_error_internals",
               reason = "better expressed through extensible enums that this \
                         enum cannot be exhaustively matched against",
               issue = "0")]
    #[doc(hidden)]
    __Nonexhaustive,
}

impl ErrorKind {
    fn as_str(&self) -> &'static str {
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
            ErrorKind::__Nonexhaustive => unreachable!()
        }
    }
}

/// Intended for use for errors not exposed to the user, where allocating onto
/// the heap (for normal construction via Error::new) is too costly.
#[stable(feature = "io_error_from_errorkind", since = "1.14.0")]
impl From<ErrorKind> for Error {
    fn from(kind: ErrorKind) -> Error {
        Error {
            repr: Repr::Simple(kind)
        }
    }
}

impl Error {
    /// Creates a new I/O error from a known kind of error as well as an
    /// arbitrary error payload.
    ///
    /// This function is used to generically create I/O errors which do not
    /// originate from the OS itself. The `error` argument is an arbitrary
    /// payload which will be contained in this `Error`.
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
        where E: Into<Box<error::Error+Send+Sync>>
    {
        Self::_new(kind, error.into())
    }

    fn _new(kind: ErrorKind, error: Box<error::Error+Send+Sync>) -> Error {
        Error {
            repr: Repr::Custom(Box::new(Custom {
                kind: kind,
                error: error,
            }))
        }
    }

    /// Returns an error representing the last OS error which occurred.
    ///
    /// This function reads the value of `errno` for the target platform (e.g.
    /// `GetLastError` on Windows) and will return a corresponding instance of
    /// `Error` for the error code.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::io::Error;
    ///
    /// println!("last OS error: {:?}", Error::last_os_error());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn last_os_error() -> Error {
        Error::from_raw_os_error(sys::os::errno() as i32)
    }

    /// Creates a new instance of an `Error` from a particular OS error code.
    ///
    /// # Examples
    ///
    /// On Linux:
    ///
    /// ```
    /// # if cfg!(target_os = "linux") {
    /// use std::io;
    ///
    /// let error = io::Error::from_raw_os_error(98);
    /// assert_eq!(error.kind(), io::ErrorKind::AddrInUse);
    /// # }
    /// ```
    ///
    /// On Windows:
    ///
    /// ```
    /// # if cfg!(windows) {
    /// use std::io;
    ///
    /// let error = io::Error::from_raw_os_error(10048);
    /// assert_eq!(error.kind(), io::ErrorKind::AddrInUse);
    /// # }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn from_raw_os_error(code: i32) -> Error {
        Error { repr: Repr::Os(code) }
    }

    /// Returns the OS error that this error represents (if any).
    ///
    /// If this `Error` was constructed via `last_os_error` or
    /// `from_raw_os_error`, then this function will return `Some`, otherwise
    /// it will return `None`.
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
    pub fn raw_os_error(&self) -> Option<i32> {
        match self.repr {
            Repr::Os(i) => Some(i),
            Repr::Custom(..) => None,
            Repr::Simple(..) => None,
        }
    }

    /// Returns a reference to the inner error wrapped by this error (if any).
    ///
    /// If this `Error` was constructed via `new` then this function will
    /// return `Some`, otherwise it will return `None`.
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
    pub fn get_ref(&self) -> Option<&(error::Error+Send+Sync+'static)> {
        match self.repr {
            Repr::Os(..) => None,
            Repr::Simple(..) => None,
            Repr::Custom(ref c) => Some(&*c.error),
        }
    }

    /// Returns a mutable reference to the inner error wrapped by this error
    /// (if any).
    ///
    /// If this `Error` was constructed via `new` then this function will
    /// return `Some`, otherwise it will return `None`.
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
    /// impl error::Error for MyError {
    ///     fn description(&self) -> &str { &self.v }
    /// }
    ///
    /// impl Display for MyError {
    ///     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
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
    pub fn get_mut(&mut self) -> Option<&mut (error::Error+Send+Sync+'static)> {
        match self.repr {
            Repr::Os(..) => None,
            Repr::Simple(..) => None,
            Repr::Custom(ref mut c) => Some(&mut *c.error),
        }
    }

    /// Consumes the `Error`, returning its inner error (if any).
    ///
    /// If this `Error` was constructed via `new` then this function will
    /// return `Some`, otherwise it will return `None`.
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
    pub fn into_inner(self) -> Option<Box<error::Error+Send+Sync>> {
        match self.repr {
            Repr::Os(..) => None,
            Repr::Simple(..) => None,
            Repr::Custom(c) => Some(c.error)
        }
    }

    /// Returns the corresponding `ErrorKind` for this error.
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
    ///     // Will print "No inner error".
    ///     print_error(Error::last_os_error());
    ///     // Will print "Inner error: ...".
    ///     print_error(Error::new(ErrorKind::AddrInUse, "oh no!"));
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn kind(&self) -> ErrorKind {
        match self.repr {
            Repr::Os(code) => sys::decode_error_kind(code),
            Repr::Custom(ref c) => c.kind,
            Repr::Simple(kind) => kind,
        }
    }
}

impl fmt::Debug for Repr {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Repr::Os(ref code) =>
                fmt.debug_struct("Os").field("code", code)
                   .field("message", &sys::os::error_string(*code)).finish(),
            Repr::Custom(ref c) => fmt.debug_tuple("Custom").field(c).finish(),
            Repr::Simple(kind) => fmt.debug_tuple("Kind").field(&kind).finish(),
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Display for Error {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match self.repr {
            Repr::Os(code) => {
                let detail = sys::os::error_string(code);
                write!(fmt, "{} (os error {})", detail, code)
            }
            Repr::Custom(ref c) => c.error.fmt(fmt),
            Repr::Simple(kind) => write!(fmt, "{}", kind.as_str()),
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl error::Error for Error {
    fn description(&self) -> &str {
        match self.repr {
            Repr::Os(..) | Repr::Simple(..) => self.kind().as_str(),
            Repr::Custom(ref c) => c.error.description(),
        }
    }

    fn cause(&self) -> Option<&error::Error> {
        match self.repr {
            Repr::Os(..) => None,
            Repr::Simple(..) => None,
            Repr::Custom(ref c) => c.error.cause(),
        }
    }
}

fn _assert_error_is_sync_send() {
    fn _is_sync_send<T: Sync+Send>() {}
    _is_sync_send::<Error>();
}

#[cfg(test)]
mod test {
    use super::{Error, ErrorKind};
    use error;
    use fmt;
    use sys::os::error_string;

    #[test]
    fn test_debug_error() {
        let code = 6;
        let msg = error_string(code);
        let err = Error { repr: super::Repr::Os(code) };
        let expected = format!("Error {{ repr: Os {{ code: {:?}, message: {:?} }} }}", code, msg);
        assert_eq!(format!("{:?}", err), expected);
    }

    #[test]
    fn test_downcasting() {
        #[derive(Debug)]
        struct TestError;

        impl fmt::Display for TestError {
            fn fmt(&self, _: &mut fmt::Formatter) -> fmt::Result {
                Ok(())
            }
        }

        impl error::Error for TestError {
            fn description(&self) -> &str {
                "asdf"
            }
        }

        // we have to call all of these UFCS style right now since method
        // resolution won't implicitly drop the Send+Sync bounds
        let mut err = Error::new(ErrorKind::Other, TestError);
        assert!(err.get_ref().unwrap().is::<TestError>());
        assert_eq!("asdf", err.get_ref().unwrap().description());
        assert!(err.get_mut().unwrap().is::<TestError>());
        let extracted = err.into_inner().unwrap();
        extracted.downcast::<TestError>().unwrap();
    }
}
