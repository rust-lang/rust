use rustc_span::ErrorGuaranteed;
use std::{env, error, fmt, fs, io};

use rustc_session::EarlyErrorHandler;

fn arg_expand(arg: &str) -> Result<Vec<String>, Error> {
    if let Some(path) = arg.strip_prefix('@') {
        let file = match fs::read_to_string(path) {
            Ok(file) => file,
            Err(ref err) if err.kind() == io::ErrorKind::InvalidData => {
                return Err(Error::Utf8Error(path.to_string()));
            }
            Err(err) => return Err(Error::IOError(path.to_string(), err)),
        };
        Ok(file.lines().map(ToString::to_string).collect())
    } else {
        Ok(vec![arg.to_string()])
    }
}

/// Replaces any `@file` arguments with the contents of `file`, with each line of `file` as a
/// separate argument.
///
/// **Note:** This function doesn't interpret argument 0 in any special way.
/// If this function is intended to be used with command line arguments,
/// `argv[0]` must be removed prior to calling it manually.
pub fn arg_expand_all(
    handler: &EarlyErrorHandler,
    at_args: &[String],
) -> Result<Vec<String>, ErrorGuaranteed> {
    let mut res = Ok(Vec::new());
    for arg in at_args {
        match arg_expand(arg) {
            Ok(arg) => {
                if let Ok(args) = &mut res {
                    args.extend(arg)
                }
            }
            Err(err) => {
                res =
                    Err(handler
                        .early_error_no_abort(format!("failed to load argument file: {err}")))
            }
        }
    }
    res
}

/// Gets the raw unprocessed command-line arguments as Unicode strings, without doing any further
/// processing (e.g., without `@file` expansion).
///
/// This function is identical to [`env::args()`] except that it emits an error when it encounters
/// non-Unicode arguments instead of panicking.
pub fn raw_args(handler: &EarlyErrorHandler) -> Result<Vec<String>, ErrorGuaranteed> {
    let mut res = Ok(Vec::new());
    for (i, arg) in env::args_os().enumerate() {
        match arg.into_string() {
            Ok(arg) => {
                if let Ok(args) = &mut res {
                    args.push(arg);
                }
            }
            Err(arg) => {
                res = Err(handler
                    .early_error_no_abort(format!("argument {i} is not valid Unicode: {arg:?}")))
            }
        }
    }
    res
}

#[derive(Debug)]
enum Error {
    Utf8Error(String),
    IOError(String, io::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Utf8Error(path) => write!(fmt, "UTF-8 error in {path}"),
            Error::IOError(path, err) => write!(fmt, "IO error: {path}: {err}"),
        }
    }
}

impl error::Error for Error {}
