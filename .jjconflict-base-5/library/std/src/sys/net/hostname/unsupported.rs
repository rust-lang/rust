use crate::ffi::OsString;
use crate::io::{Error, Result};

pub fn hostname() -> Result<OsString> {
    Err(Error::UNSUPPORTED_PLATFORM)
}
