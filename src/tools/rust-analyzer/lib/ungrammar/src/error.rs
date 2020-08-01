//! Boilerplate error definitions.
use std::fmt;

use crate::lexer::Location;

pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug)]
pub struct Error {
    pub(crate) message: String,
    pub(crate) location: Option<Location>,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(loc) = self.location {
            write!(f, "{}:{}: ", loc.line, loc.column)?
        }
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for Error {}

impl Error {
    pub(crate) fn with_location(self, location: Location) -> Error {
        Error {
            location: Some(location),
            ..self
        }
    }
}

macro_rules! _format_err {
    ($($tt:tt)*) => {
        $crate::error::Error {
            message: format!($($tt)*),
            location: None,
        }
    };
}
pub(crate) use _format_err as format_err;

macro_rules! _bail {
    ($($tt:tt)*) => { return Err($crate::error::format_err!($($tt)*)) };
}
pub(crate) use _bail as bail;
