use std::error;
use std::fmt;
use std::fs;
use std::io;

fn arg_expand(arg: String) -> Result<Vec<String>, Error> {
    if let Some(path) = arg.strip_prefix('@') {
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

/// **Note:** This function doesn't interpret argument 0 in any special way.
/// If this function is intended to be used with command line arguments,
/// `argv[0]` must be removed prior to calling it manually.
pub fn arg_expand_all(at_args: &[String]) -> Vec<String> {
    let mut args = Vec::new();
    for arg in at_args {
        match arg_expand(arg.clone()) {
            Ok(arg) => args.extend(arg),
            Err(err) => rustc_session::early_error(
                rustc_session::config::ErrorOutputType::default(),
                format!("Failed to load argument file: {err}"),
            ),
        }
    }
    args
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
            Error::Utf8Error(Some(path)) => write!(fmt, "Utf8 error in {path}"),
            Error::IOError(path, err) => write!(fmt, "IO Error: {path}: {err}"),
        }
    }
}

impl error::Error for Error {}
