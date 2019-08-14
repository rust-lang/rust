use std::env;
use std::error;
use std::fmt;
use std::fs;
use std::io;
use std::str;
use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(test)]
mod tests;

static USED_ARGSFILE_FEATURE: AtomicBool = AtomicBool::new(false);

pub fn used_unstable_argsfile() -> bool {
    USED_ARGSFILE_FEATURE.load(Ordering::Relaxed)
}

pub struct ArgsIter {
    base: env::ArgsOs,
    file: std::vec::IntoIter<String>,
}

impl ArgsIter {
    pub fn new() -> Self {
        ArgsIter { base: env::args_os(), file: vec![].into_iter() }
    }
}

impl Iterator for ArgsIter {
    type Item = Result<String, Error>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(line) = self.file.next() {
                return Some(Ok(line));
            }

            let arg =
                self.base.next().map(|arg| arg.into_string().map_err(|_| Error::Utf8Error(None)));
            match arg {
                Some(Err(err)) => return Some(Err(err)),
                Some(Ok(ref arg)) if arg.starts_with("@") => {
                    let path = &arg[1..];
                    let file = match fs::read_to_string(path) {
                        Ok(file) => {
                            USED_ARGSFILE_FEATURE.store(true, Ordering::Relaxed);
                            file
                        }
                        Err(ref err) if err.kind() == io::ErrorKind::InvalidData => {
                            return Some(Err(Error::Utf8Error(Some(path.to_string()))));
                        }
                        Err(err) => return Some(Err(Error::IOError(path.to_string(), err))),
                    };
                    self.file =
                        file.lines().map(ToString::to_string).collect::<Vec<_>>().into_iter();
                }
                Some(Ok(arg)) => return Some(Ok(arg)),
                None => return None,
            }
        }
    }
}

#[derive(Debug)]
pub enum Error {
    Utf8Error(Option<String>),
    IOError(String, io::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Utf8Error(None) => write!(fmt, "Utf8 error"),
            Error::Utf8Error(Some(path)) => write!(fmt, "Utf8 error in {}", path),
            Error::IOError(path, err) => write!(fmt, "IO Error: {}: {}", path, err),
        }
    }
}

impl error::Error for Error {
    fn description(&self) -> &'static str {
        "argument error"
    }
}
