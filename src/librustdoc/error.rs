use std::error;
use std::fmt::{self, Formatter};
use std::path::{Path, PathBuf};

use crate::docfs::PathError;

#[derive(Debug)]
crate struct Error {
    crate file: PathBuf,
    crate error: String,
}

impl error::Error for Error {}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let file = self.file.display().to_string();
        if file.is_empty() {
            write!(f, "{}", self.error)
        } else {
            write!(f, "\"{}\": {}", self.file.display(), self.error)
        }
    }
}

impl PathError for Error {
    fn new<S, P: AsRef<Path>>(e: S, path: P) -> Error
    where
        S: ToString + Sized,
    {
        Error { file: path.as_ref().to_path_buf(), error: e.to_string() }
    }
}

#[macro_export]
macro_rules! try_none {
    ($e:expr, $file:expr) => {{
        use std::io;
        match $e {
            Some(e) => e,
            None => {
                return Err(Error::new(io::Error::new(io::ErrorKind::Other, "not found"), $file));
            }
        }
    }};
}

#[macro_export]
macro_rules! try_err {
    ($e:expr, $file:expr) => {{
        match $e {
            Ok(e) => e,
            Err(e) => return Err(Error::new(e, $file)),
        }
    }};
}
