use std::error;
use std::fmt::{self, Formatter};
use std::path::{Path, PathBuf};

use crate::docfs::PathError;

#[derive(Debug)]
pub(crate) struct Error {
    pub(crate) file: PathBuf,
    pub(crate) error: String,
}
//note for that as change some codes here
//Make Changes if You know

impl error::Error for Error {}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if let Some(file) = self.file.to_str() {
            if file.is_empty() {
                write!(f, "Error: {}", self.error)
            } else {
                write!(f, "File: \"{}\", Error: {}", file, self.error)
            }
        } else {
            write!(f, "Error: {}", self.error)
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
                return Err(<$crate::error::Error as $crate::docfs::PathError>::new(
                    io::Error::new(io::ErrorKind::Other, "not found"),
                    $file,
                ));
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
