use std::env;
use std::ffi::{OsStr, OsString};
use std::io::Write;
use std::path::Path;
use std::process::{Command, CommandEnvs};

use camino::{Utf8Path, Utf8PathBuf};
use tempfile::NamedTempFile;

#[cfg(test)]
mod tests;

pub(crate) fn make_new_path(path: &str) -> String {
    assert!(cfg!(windows));
    // Windows just uses PATH as the library search path, so we have to
    // maintain the current value while adding our own
    match env::var(lib_path_env_var()) {
        Ok(curr) => format!("{}{}{}", path, path_div(), curr),
        Err(..) => path.to_owned(),
    }
}

pub(crate) fn lib_path_env_var() -> &'static str {
    "PATH"
}
fn path_div() -> &'static str {
    ";"
}

pub(crate) trait Utf8PathBufExt {
    /// Append an extension to the path, even if it already has one.
    fn with_extra_extension(&self, extension: &str) -> Utf8PathBuf;
}

impl Utf8PathBufExt for Utf8PathBuf {
    fn with_extra_extension(&self, extension: &str) -> Utf8PathBuf {
        if extension.is_empty() {
            self.clone()
        } else {
            let mut fname = self.file_name().unwrap().to_string();
            if !extension.starts_with('.') {
                fname.push_str(".");
            }
            fname.push_str(extension);
            self.with_file_name(fname)
        }
    }
}

/// The name of the environment variable that holds dynamic library locations.
pub(crate) fn dylib_env_var() -> &'static str {
    if cfg!(any(windows, target_os = "cygwin")) {
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

/// Adds a list of lookup paths to `cmd`'s dynamic library lookup path.
/// If the dylib_path_var is already set for this cmd, the old value will be overwritten!
pub(crate) fn add_dylib_path(
    cmd: &mut Command,
    paths: impl Iterator<Item = impl Into<std::path::PathBuf>>,
) {
    let path_env = env::var_os(dylib_env_var());
    let old_paths = path_env.as_ref().map(env::split_paths);
    let new_paths = paths.map(Into::into).chain(old_paths.into_iter().flatten());
    cmd.env(dylib_env_var(), env::join_paths(new_paths).unwrap());
}

pub(crate) fn copy_dir_all(src: &Utf8Path, dst: &Utf8Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dst.as_std_path())?;
    for entry in std::fs::read_dir(src.as_std_path())? {
        let entry = entry?;
        let path = Utf8PathBuf::try_from(entry.path()).unwrap();
        let file_name = path.file_name().unwrap();
        let ty = entry.file_type()?;
        if ty.is_dir() {
            copy_dir_all(&path, &dst.join(file_name))?;
        } else {
            std::fs::copy(path.as_std_path(), dst.join(file_name).as_std_path())?;
        }
    }
    Ok(())
}

macro_rules! static_regex {
    ($re:literal) => {{
        static RE: ::std::sync::OnceLock<::regex::Regex> = ::std::sync::OnceLock::new();
        RE.get_or_init(|| ::regex::Regex::new($re).unwrap())
    }};
}
pub(crate) use static_regex;

macro_rules! string_enum {
    (
        $(#[$meta:meta])*
        $vis:vis enum $name:ident {
            $(
                $(#[$variant_meta:meta])*
                $variant:ident => $repr:expr,
            )*
        }
    ) => {
        $(#[$meta])*
        $vis enum $name {
            $(
                $(#[$variant_meta])*
                $variant,
            )*
        }

        impl $name {
            #[allow(dead_code)]
            $vis const VARIANTS: &'static [Self] = &[
                $( Self::$variant, )*
            ];
            #[allow(dead_code)]
            $vis const STR_VARIANTS: &'static [&'static str] = &[
                $( Self::$variant.to_str(), )*
            ];

            $vis const fn to_str(&self) -> &'static str {
                match self {
                    $( Self::$variant => $repr, )*
                }
            }
        }

        impl ::std::fmt::Display for $name {
            fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                ::std::fmt::Display::fmt(self.to_str(), f)
            }
        }

        impl ::std::str::FromStr for $name {
            type Err = String;

            fn from_str(s: &str) -> Result<Self, Self::Err> {
                match s {
                    $( $repr => Ok(Self::$variant), )*
                    _ => Err(format!(concat!("unknown `", stringify!($name), "` variant: `{}`"), s)),
                }
            }
        }
    }
}

pub(crate) use string_enum;

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
pub(crate) struct ArgFileCommand {
    command: Command,
    args: Vec<OsString>,
}

#[allow(dead_code)] // Roughly match the `std::process::Command` API
impl ArgFileCommand {
    #[track_caller]
    pub(crate) fn new<S: AsRef<OsStr>>(program: S) -> Self {
        let command = Command::new(program);
        Self { command, args: Vec::new() }
    }
    pub(crate) fn arg<S: AsRef<OsStr>>(&mut self, arg: S) -> &mut Self {
        self.args.push(arg.as_ref().to_os_string());
        self
    }

    pub(crate) fn args<I, S>(&mut self, args: I) -> &mut Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
    {
        self.args.extend(args.into_iter().map(|s| s.as_ref().to_os_string()));
        self
    }

    pub(crate) fn env<K, V>(&mut self, key: K, val: V) -> &mut Self
    where
        K: AsRef<OsStr>,
        V: AsRef<OsStr>,
    {
        self.command.env(key, val);
        self
    }

    pub(crate) fn get_envs(&self) -> CommandEnvs<'_> {
        self.command.get_envs()
    }

    pub(crate) fn env_remove<K: AsRef<OsStr>>(&mut self, key: K) -> &mut Self {
        self.command.env_remove(key);
        self
    }

    pub(crate) fn current_dir<P: AsRef<Path>>(&mut self, dir: P) -> &mut Self {
        self.command.current_dir(dir);
        self
    }

    pub(crate) fn stdin(&mut self, stdin: std::process::Stdio) -> &mut Self {
        self.command.stdin(stdin);
        self
    }

    pub(crate) fn build(mut self) -> std::io::Result<(Command, Option<NamedTempFile>)> {
        // On Windows there is a hard limit of ~32KB, so we cut off at 30KB to
        // give some buffer just in case.
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

        let mut tmp = tempfile::Builder::new().prefix("compiletest-argfile.").tempfile()?;

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
