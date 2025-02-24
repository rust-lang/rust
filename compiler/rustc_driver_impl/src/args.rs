use std::{env, error, fmt, fs, io};

use rustc_session::EarlyDiagCtxt;

/// Expands argfiles in command line arguments.
#[derive(Default)]
struct Expander {
    shell_argfiles: bool,
    next_is_unstable_option: bool,
    expanded: Vec<String>,
}

impl Expander {
    /// Handles the next argument. If the argument is an argfile, it is expanded
    /// inline.
    fn arg(&mut self, arg: &str) -> Result<(), Error> {
        if let Some(argfile) = arg.strip_prefix('@') {
            match argfile.split_once(':') {
                Some(("shell", path)) if self.shell_argfiles => {
                    shlex::split(&Self::read_file(path)?)
                        .ok_or_else(|| Error::ShellParseError(path.to_string()))?
                        .into_iter()
                        .for_each(|arg| self.push(arg));
                }
                _ => {
                    let contents = Self::read_file(argfile)?;
                    contents.lines().for_each(|arg| self.push(arg.to_string()));
                }
            }
        } else {
            self.push(arg.to_string());
        }

        Ok(())
    }

    /// Adds a command line argument verbatim with no argfile expansion.
    fn push(&mut self, arg: String) {
        // Unfortunately, we have to do some eager argparsing to handle unstable
        // options which change the behavior of argfile arguments.
        //
        // Normally, all of the argfile arguments (e.g. `@args.txt`) are
        // expanded into our arguments list *and then* the whole list of
        // arguments are passed on to be parsed. However, argfile parsing
        // options like `-Zshell_argfiles` need to change the behavior of that
        // argument expansion. So we have to do a little parsing on our own here
        // instead of leaning on the existing logic.
        //
        // All we care about are unstable options, so we parse those out and
        // look for any that affect how we expand argfiles. This argument
        // inspection is very conservative; we only change behavior when we see
        // exactly the options we're looking for and everything gets passed
        // through.

        if self.next_is_unstable_option {
            self.inspect_unstable_option(&arg);
            self.next_is_unstable_option = false;
        } else if let Some(unstable_option) = arg.strip_prefix("-Z") {
            if unstable_option.is_empty() {
                self.next_is_unstable_option = true;
            } else {
                self.inspect_unstable_option(unstable_option);
            }
        }

        self.expanded.push(arg);
    }

    /// Consumes the `Expander`, returning the expanded arguments.
    fn finish(self) -> Vec<String> {
        self.expanded
    }

    /// Parses any relevant unstable flags specified on the command line.
    fn inspect_unstable_option(&mut self, option: &str) {
        match option {
            "shell-argfiles" => self.shell_argfiles = true,
            _ => (),
        }
    }

    /// Reads the contents of a file as UTF-8.
    fn read_file(path: &str) -> Result<String, Error> {
        fs::read_to_string(path).map_err(|e| {
            if e.kind() == io::ErrorKind::InvalidData {
                Error::Utf8Error(path.to_string())
            } else {
                Error::IOError(path.to_string(), e)
            }
        })
    }
}

/// Replaces any `@file` arguments with the contents of `file`, with each line of `file` as a
/// separate argument.
///
/// **Note:** This function doesn't interpret argument 0 in any special way.
/// If this function is intended to be used with command line arguments,
/// `argv[0]` must be removed prior to calling it manually.
#[allow(rustc::untranslatable_diagnostic)] // FIXME: make this translatable
pub fn arg_expand_all(early_dcx: &EarlyDiagCtxt, at_args: &[String]) -> Vec<String> {
    let mut expander = Expander::default();
    let mut result = Ok(());
    for arg in at_args {
        if let Err(err) = expander.arg(arg) {
            result = Err(early_dcx.early_err(format!("failed to load argument file: {err}")));
        }
    }
    if let Err(guar) = result {
        guar.raise_fatal();
    }
    expander.finish()
}

/// Gets the raw unprocessed command-line arguments as Unicode strings, without doing any further
/// processing (e.g., without `@file` expansion).
///
/// This function is identical to [`env::args()`] except that it emits an error when it encounters
/// non-Unicode arguments instead of panicking.
pub fn raw_args(early_dcx: &EarlyDiagCtxt) -> Vec<String> {
    let mut args = Vec::new();
    let mut guar = Ok(());
    for (i, arg) in env::args_os().enumerate() {
        match arg.into_string() {
            Ok(arg) => args.push(arg),
            Err(arg) => {
                guar =
                    Err(early_dcx.early_err(format!("argument {i} is not valid Unicode: {arg:?}")))
            }
        }
    }
    if let Err(guar) = guar {
        guar.raise_fatal();
    }
    args
}

#[derive(Debug)]
enum Error {
    Utf8Error(String),
    IOError(String, io::Error),
    ShellParseError(String),
}

impl fmt::Display for Error {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Utf8Error(path) => write!(fmt, "UTF-8 error in {path}"),
            Error::IOError(path, err) => write!(fmt, "IO error: {path}: {err}"),
            Error::ShellParseError(path) => write!(fmt, "invalid shell-style arguments in {path}"),
        }
    }
}

impl error::Error for Error {}
