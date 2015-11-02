use fmt;
use sys::error::Error;

pub type Result<T> = ::result::Result<T, Error>;

pub fn expect_last_result<T>() -> Result<T> {
    Err(expect_last_error())
}

pub fn expect_last_error() -> Error {
    Error::last_error().unwrap_or_else(|| Error::from_code(0))
}

impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&(self.code(), self.description().to_string_lossy()), f)
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&self.description().to_string_lossy(), f)
    }
}
