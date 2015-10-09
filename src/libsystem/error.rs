pub use imp::error as imp;

pub mod traits {
    pub use error::{Error as sys_Error, ErrorString as sys_ErrorString};
}

pub mod prelude {
    pub use super::imp::Error;
    pub use super::traits::*;

    pub type ErrorString = <Error as sys_Error>::ErrorString;
    pub type Result<T> = ::core::result::Result<T, Error>;
}

use core::str;
use core::fmt;
use core::result;
use collections::borrow::Cow;
use collections::String;

pub trait Error: fmt::Debug + fmt::Display + From<fmt::Error> {
    type ErrorString: ErrorString;

    fn from_code(code: i32) -> Self where Self: Sized;
    fn last_error() -> Option<Self> where Self: Sized;

    fn expect_last_result<T>() -> Result<T, Self> where Self: Sized {
        Err(Self::expect_last_error())
    }

    fn expect_last_error() -> Self where Self: Sized {
        Self::last_error().unwrap_or_else(|| Self::from_code(-1))
    }

    fn code(&self) -> i32;
    fn description(&self) -> Self::ErrorString;
}

pub trait ErrorString: Sized {
    fn as_bytes(&self) -> &[u8];

    fn to_str(&self) -> Result<&str, str::Utf8Error> {
        str::from_utf8(self.as_bytes())
    }

    fn to_string_lossy(&self) -> Cow<str> {
        String::from_utf8_lossy(self.as_bytes())
    }
}
