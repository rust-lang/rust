#![allow(dead_code)]

use std::env;
use std::error;
use std::fmt;
use std::fs;
use std::io;
use std::str;

#[cfg(test)]
mod tests;

/// States for parsing text
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum State {
    Normal, // within normal text
    Cr,     // just saw \r
    Lf,     // just saw \n
}

struct FileArgs {
    path: String,
    input: Vec<u8>,
    offset: usize,
}

impl FileArgs {
    pub fn new(path: String, input: Vec<u8>) -> Self {
        FileArgs { path, input, offset: 0 }
    }
}

impl Iterator for FileArgs {
    type Item = Result<String, Error>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.offset >= self.input.len() {
            // All done
            return None;
        }

        use State::*;
        let mut state = Normal;
        let start = self.offset;
        let mut end = start;

        for (idx, b) in self.input[start..].iter().enumerate() {
            let idx = start + idx + 1;

            self.offset = idx;

            match (b, state) {
                (b'\r', Normal) => state = Cr,
                (b'\n', Normal) => state = Lf,

                (b'\r', Lf) | (b'\n', Cr) => {
                    // Two-character line break (accept \r\n and \n\r(?)), so consume them both
                    break;
                }

                (_, Cr) | (_, Lf) => {
                    // Peeked at character after single-character line break, so rewind to visit it
                    // next time around.
                    self.offset = idx - 1;
                    break;
                }

                (_, _) => {
                    end = idx;
                    state = Normal;
                }
            }
        }

        Some(
            String::from_utf8(self.input[start..end].to_vec())
                .map_err(|_| Error::Utf8Error(Some(self.path.clone()))),
        )
    }
}

pub struct ArgsIter {
    base: env::ArgsOs,
    file: Option<FileArgs>,
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
                    // can't not be utf-8 now
                    let path = str::from_utf8(&arg.as_bytes()[1..]).unwrap();
                    let file = match fs::read(path) {
                        Ok(file) => file,
                        Err(err) => return Some(Err(Error::IOError(path.to_string(), err))),
                    };
                    self.file = Some(FileArgs::new(path.to_string(), file));
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
