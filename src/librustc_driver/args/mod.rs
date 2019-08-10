use std::env;
use std::error;
use std::fmt;
use std::fs;
use std::io::{self, BufRead};
use std::str;
use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(test)]
mod tests;

static USED_ARGSFILE_FEATURE: AtomicBool = AtomicBool::new(false);

pub fn used_unstable_argsfile() -> bool {
    USED_ARGSFILE_FEATURE.load(Ordering::Relaxed)
}

struct FileArgs {
    path: String,
    input: Vec<u8>,
}

impl FileArgs {
    fn new(path: String, input: Vec<u8>) -> Self {
        FileArgs { path, input }
    }

    fn lines(self) -> impl Iterator<Item = Result<String, Error>> {
        let Self { input, path } = self;
        io::Cursor::new(input).lines().map(move |res| {
            let path = path.clone();
            res.map_err(move |err| match err.kind() {
                io::ErrorKind::InvalidData => Error::Utf8Error(Some(path)),
                _ => Error::IOError(path, err),
            })
        })
    }
}

pub struct ArgsIter {
    base: env::ArgsOs,
    file: Option<Box<dyn Iterator<Item = Result<String, Error>>>>,
}

impl ArgsIter {
    pub fn new() -> Self {
        ArgsIter { base: env::args_os(), file: None }
    }
}

impl Iterator for ArgsIter {
    type Item = Result<String, Error>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(ref mut file) = &mut self.file {
                match file.next() {
                    Some(res) => return Some(res.map_err(From::from)),
                    None => self.file = None,
                }
            }

            let arg =
                self.base.next().map(|arg| arg.into_string().map_err(|_| Error::Utf8Error(None)));
            match arg {
                Some(Err(err)) => return Some(Err(err)),
                Some(Ok(ref arg)) if arg.starts_with("@") => {
                    let path = &arg[1..];
                    let lines = match fs::read(path) {
                        Ok(file) => {
                            USED_ARGSFILE_FEATURE.store(true, Ordering::Relaxed);
                            FileArgs::new(path.to_string(), file).lines()
                        }
                        Err(err) => return Some(Err(Error::IOError(path.to_string(), err))),
                    };
                    self.file = Some(Box::new(lines));
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
