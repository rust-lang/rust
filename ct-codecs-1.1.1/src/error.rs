use core::fmt::{self, Display};

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Error {
    /// The provided output buffer would be too small.
    Overflow,
    /// The input isn't valid for the given encoding.
    InvalidInput,
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}

impl Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Overflow => write!(f, "Overflow"),
            Error::InvalidInput => write!(f, "Invalid input"),
        }
    }
}
