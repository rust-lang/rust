//! When things go wrong, we need some error handling.
//! There are a few different types of errors in StableMIR:
//!
//! - [CompilerError]: This represents errors that can be raised when invoking the compiler.
//! - [Error]: Generic error that represents the reason why a request that could not be fulfilled.

use std::convert::From;
use std::fmt::{Debug, Display, Formatter};
use std::{error, fmt};

/// An error type used to represent an error that has already been reported by the compiler.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum CompilerError<T> {
    /// Internal compiler error (I.e.: Compiler crashed).
    ICE,
    /// Compilation failed.
    CompilationFailed,
    /// Compilation was interrupted.
    Interrupted(T),
    /// Compilation skipped. This happens when users invoke rustc to retrieve information such as
    /// --version.
    Skipped,
}

/// A generic error to represent an API request that cannot be fulfilled.
#[derive(Debug)]
pub struct Error(String);

impl Error {
    pub(crate) fn new(msg: String) -> Self {
        Self(msg)
    }
}

impl From<&str> for Error {
    fn from(value: &str) -> Self {
        Self(value.into())
    }
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.0, f)
    }
}

impl<T> Display for CompilerError<T>
where
    T: Display,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            CompilerError::ICE => write!(f, "Internal Compiler Error"),
            CompilerError::CompilationFailed => write!(f, "Compilation Failed"),
            CompilerError::Interrupted(reason) => write!(f, "Compilation Interrupted: {reason}"),
            CompilerError::Skipped => write!(f, "Compilation Skipped"),
        }
    }
}

impl<T> Debug for CompilerError<T>
where
    T: Debug,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            CompilerError::ICE => write!(f, "Internal Compiler Error"),
            CompilerError::CompilationFailed => write!(f, "Compilation Failed"),
            CompilerError::Interrupted(reason) => write!(f, "Compilation Interrupted: {reason:?}"),
            CompilerError::Skipped => write!(f, "Compilation Skipped"),
        }
    }
}

impl error::Error for Error {}
impl<T> error::Error for CompilerError<T> where T: Display + Debug {}
