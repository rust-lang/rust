//! This module serves two purposes:
//!
//! 1. It is part of the `utils` module and used in other parts of bootstrap.
//! 2. It is embedded inside bootstrap shims to avoid a dependency on the bootstrap library.
//!    Therefore, this module should never use any other bootstrap module. This reduces binary size
//!    and improves compilation time by minimizing linking time.

// # Note on tests
//
// If we were to declare a tests submodule here, the shim binaries that include this module via
// `#[path]` would fail to find it, which breaks `./x check bootstrap`. So instead the unit tests
// for this module are in `super::tests::shared_helpers_tests`.

#![allow(dead_code)]

use std::env;
use std::ffi::{OsStr, OsString};
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::process::{Command, CommandEnvs};
use std::str::FromStr;

use tempfile::NamedTempFile;

/// Returns the environment variable which the dynamic library lookup path
/// resides in for this platform.
pub fn dylib_path_var() -> &'static str {
    if cfg!(any(target_os = "windows", target_os = "cygwin")) {
        "PATH"
    } else if cfg!(target_vendor = "apple") {
        "DYLD_LIBRARY_PATH"
    } else if cfg!(target_os = "haiku") {
        "LIBRARY_PATH"
    } else if cfg!(target_os = "aix") {
        "LIBPATH"
    } else {
        "LD_LIBRARY_PATH"
    }
}

/// Parses the `dylib_path_var()` environment variable, returning a list of
/// paths that are members of this lookup path.
pub fn dylib_path() -> Vec<std::path::PathBuf> {
    let var = match std::env::var_os(dylib_path_var()) {
        Some(v) => v,
        None => return vec![],
    };
    std::env::split_paths(&var).collect()
}

/// Given an executable called `name`, return the filename for the
/// executable for a particular target.
pub fn exe(name: &str, target: &str) -> String {
    // On Cygwin, the decision to append .exe or not is not as straightforward.
    // Executable files do actually have .exe extensions so on hosts other than
    // Cygwin it is necessary.  But on a Cygwin host there is magic happening
    // that redirects requests for file X to file X.exe if it exists, and
    // furthermore /proc/self/exe (and thus std::env::current_exe) always
    // returns the name *without* the .exe extension.  For comparisons against
    // that to match, we therefore do not append .exe for Cygwin targets on
    // a Cygwin host.
    if target.contains("windows") || (cfg!(not(target_os = "cygwin")) && target.contains("cygwin"))
    {
        format!("{name}.exe")
    } else if target.contains("uefi") {
        format!("{name}.efi")
    } else if target.contains("wasm") {
        format!("{name}.wasm")
    } else {
        name.to_string()
    }
}

/// Parses the value of the "RUSTC_VERBOSE" environment variable and returns it as a `usize`.
/// If it was not defined, returns 0 by default.
///
/// Panics if "RUSTC_VERBOSE" is defined with the value that is not an unsigned integer.
pub fn parse_rustc_verbose() -> usize {
    match env::var("RUSTC_VERBOSE") {
        Ok(s) => usize::from_str(&s).expect("RUSTC_VERBOSE should be an integer"),
        Err(_) => 0,
    }
}

/// Parses the value of the "RUSTC_STAGE" environment variable and returns it as a `String`.
/// This is the stage of the *build compiler*, which we are wrapping using a rustc/rustdoc wrapper.
///
/// If "RUSTC_STAGE" was not set, the program will be terminated with 101.
pub fn parse_rustc_stage() -> u32 {
    env::var("RUSTC_STAGE").ok().and_then(|v| v.parse().ok()).unwrap_or_else(|| {
        // Don't panic here; it's reasonable to try and run these shims directly. Give a helpful error instead.
        eprintln!("rustc shim: FATAL: RUSTC_STAGE was not set");
        eprintln!("rustc shim: NOTE: use `x.py build -vvv` to see all environment variables set by bootstrap");
        std::process::exit(101);
    })
}

/// Writes the command invocation to a file if `DUMP_BOOTSTRAP_SHIMS` is set during bootstrap.
///
/// Before writing it, replaces user-specific values to create generic dumps for cross-environment
/// comparisons.
pub fn maybe_dump(dump_name: String, cmd: &Command) {
    if let Ok(dump_dir) = env::var("DUMP_BOOTSTRAP_SHIMS") {
        let dump_file = format!("{dump_dir}/{dump_name}");

        let mut file = OpenOptions::new().create(true).append(true).open(dump_file).unwrap();

        let cmd_dump = format!("{cmd:?}\n");
        let cmd_dump = cmd_dump.replace(&env::var("BUILD_OUT").unwrap(), "${BUILD_OUT}");
        let cmd_dump = cmd_dump.replace(&env::var("CARGO_HOME").unwrap(), "${CARGO_HOME}");

        file.write_all(cmd_dump.as_bytes()).expect("Unable to write file");
    }
}

/// Finds `key` and returns its value from the given list of arguments `args`.
pub fn parse_value_from_args<'a>(args: &'a [OsString], key: &str) -> Option<&'a str> {
    let mut args = args.iter();
    while let Some(arg) = args.next() {
        let arg = arg.to_str().unwrap();

        if let Some(value) = arg.strip_prefix(&format!("{key}=")) {
            return Some(value);
        } else if arg == key {
            return args.next().map(|v| v.to_str().unwrap());
        }
    }

    None
}

/// A wrapper around [`Command`] that adds support for arg files.
/// This is useful as we have some commands that can get very long and at times
/// hit the OS limit (usually Windows)
///
/// This implementation is based off the of `ProcessBuilder` implementation in Cargo
/// but simplified.
///
/// NOTE: In most scenarios we want to avoid arg files as it makes debugging more complicated
///       so we try to avoid it if the command is not close the the OS limit.
#[derive(Debug)]
pub struct ArgFileCommand {
    command: Command,
    args: Vec<OsString>,
}

impl ArgFileCommand {
    #[track_caller]
    pub fn new<S: AsRef<OsStr>>(program: S) -> Self {
        let command = Command::new(program);
        Self { command, args: Vec::new() }
    }
    pub fn arg<S: AsRef<OsStr>>(&mut self, arg: S) -> &mut Self {
        self.args.push(arg.as_ref().to_os_string());
        self
    }

    pub fn args<I, S>(&mut self, args: I) -> &mut Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
    {
        self.args.extend(args.into_iter().map(|s| s.as_ref().to_os_string()));
        self
    }

    pub fn env<K, V>(&mut self, key: K, val: V) -> &mut Self
    where
        K: AsRef<OsStr>,
        V: AsRef<OsStr>,
    {
        self.command.env(key, val);
        self
    }

    pub fn get_envs(&self) -> CommandEnvs<'_> {
        self.command.get_envs()
    }

    pub fn env_remove<K: AsRef<OsStr>>(&mut self, key: K) -> &mut Self {
        self.command.env_remove(key);
        self
    }

    pub fn current_dir<P: AsRef<Path>>(&mut self, dir: P) -> &mut Self {
        self.command.current_dir(dir);
        self
    }

    pub fn stdin(&mut self, stdin: std::process::Stdio) -> &mut Self {
        self.command.stdin(stdin);
        self
    }

    pub fn build(mut self) -> std::io::Result<(Command, Option<NamedTempFile>)> {
        // On Windows there is a hard limit of ~32KB, so we cut off at 30KB to
        // give some buffer just incase.
        #[cfg(windows)]
        let threshold: usize = 30 * 1024;
        // On unix the limit is defined by ARG_MAX. If its not explicitly set we set it to 1MB
        // which is fairly large but lower than the ~2MB that it defaults to on most systems.
        #[cfg(unix)]
        let threshold: usize =
            std::env::var("ARG_MAX").ok().and_then(|v| v.parse().ok()).unwrap_or(1024 * 1024);

        let total_arg_len: usize = self.args.iter().map(|a| a.len() + 1).sum();
        if total_arg_len <= threshold {
            self.command.args(self.args);
            return Ok((self.command, None));
        }

        let mut tmp = tempfile::Builder::new().prefix("bootstrap-argfile.").tempfile()?;

        let mut arg = OsString::from("@");
        arg.push(tmp.path());
        self.command.arg(arg);

        let mut buf = Vec::with_capacity(total_arg_len);
        for arg in &self.args {
            let arg = arg.to_str().ok_or_else(|| {
                std::io::Error::other(format!(
                    "argument for argfile contains invalid UTF-8 characters: `{}`",
                    arg.to_string_lossy()
                ))
            })?;
            if arg.contains('\n') {
                return Err(std::io::Error::other(format!(
                    "argument for argfile contains newlines: `{arg}`"
                )));
            }
            writeln!(buf, "{arg}")?;
        }
        tmp.write_all(&buf)?;

        Ok((self.command, Some(tmp)))
    }
}
