use std::error;
use std::fmt;
use std::fs;
use std::io;

pub fn arg_expand(arg: String) -> Result<Vec<String>, Error> {
    if arg.starts_with("@") {
        let path = &arg[1..];
        let file = match fs::read_to_string(path) {
            Ok(file) => file,
            Err(ref err) if err.kind() == io::ErrorKind::InvalidData => {
                return Err(Error::Utf8Error(Some(path.to_string())));
            }
            Err(err) => return Err(Error::IOError(path.to_string(), err)),
        };
        Ok(file.lines().map(ToString::to_string).collect())
    } else {
        Ok(vec![arg])
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

impl error::Error for Error {}
