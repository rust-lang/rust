pub use sys::imp::error::{
    Error,
    ErrorString,
};

use str;
use borrow::Cow;
use string::String;

pub type Result<T> = ::result::Result<T, Error>;

pub fn expect_last_result<T>() -> Result<T> {
    Err(expect_last_error())
}

pub fn expect_last_error() -> Error {
    Error::last_error().unwrap_or_else(|| Error::from_code(0))
}

impl ErrorString {
    pub fn to_str(&self) -> ::result::Result<&str, str::Utf8Error> {
        str::from_utf8(self.as_bytes())
    }

    pub fn to_string_lossy(&self) -> Cow<str> {
        String::from_utf8_lossy(self.as_bytes())
    }
}
