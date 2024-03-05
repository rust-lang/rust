//! Discovery of `cargo` & `rustc` executables.

#![warn(rust_2018_idioms, unused_lifetimes)]

use std::{
    env, iter,
    path::{Path, PathBuf},
};

#[derive(Copy, Clone)]
pub enum Tool {
    Cargo,
    Rustc,
    Rustup,
    Rustfmt,
}

impl Tool {
    pub fn proxy(self) -> Option<PathBuf> {
        cargo_proxy(self.name())
    }

    /// Return a `PathBuf` to use for the given executable.
    ///
    /// The current implementation checks three places for an executable to use:
    /// 1) `$CARGO_HOME/bin/<executable_name>`
    ///      where $CARGO_HOME defaults to ~/.cargo (see https://doc.rust-lang.org/cargo/guide/cargo-home.html)
    ///      example: for cargo, this tries $CARGO_HOME/bin/cargo, or ~/.cargo/bin/cargo if $CARGO_HOME is unset.
    ///      It seems that this is a reasonable place to try for cargo, rustc, and rustup
    /// 2) Appropriate environment variable (erroring if this is set but not a usable executable)
    ///      example: for cargo, this checks $CARGO environment variable; for rustc, $RUSTC; etc
    /// 3) $PATH/`<executable_name>`
    ///      example: for cargo, this tries all paths in $PATH with appended `cargo`, returning the
    ///      first that exists
    /// 4) If all else fails, we just try to use the executable name directly
    pub fn prefer_proxy(self) -> PathBuf {
        invoke(&[cargo_proxy, lookup_as_env_var, lookup_in_path], self.name())
    }

    /// Return a `PathBuf` to use for the given executable.
    ///
    /// The current implementation checks three places for an executable to use:
    /// 1) Appropriate environment variable (erroring if this is set but not a usable executable)
    ///      example: for cargo, this checks $CARGO environment variable; for rustc, $RUSTC; etc
    /// 2) $PATH/`<executable_name>`
    ///      example: for cargo, this tries all paths in $PATH with appended `cargo`, returning the
    ///      first that exists
    /// 3) `$CARGO_HOME/bin/<executable_name>`
    ///      where $CARGO_HOME defaults to ~/.cargo (see https://doc.rust-lang.org/cargo/guide/cargo-home.html)
    ///      example: for cargo, this tries $CARGO_HOME/bin/cargo, or ~/.cargo/bin/cargo if $CARGO_HOME is unset.
    ///      It seems that this is a reasonable place to try for cargo, rustc, and rustup
    /// 4) If all else fails, we just try to use the executable name directly
    pub fn path(self) -> PathBuf {
        invoke(&[lookup_as_env_var, lookup_in_path, cargo_proxy], self.name())
    }

    pub fn path_in(self, path: &Path) -> Option<PathBuf> {
        probe_for_binary(path.join(self.name()))
    }

    pub fn name(self) -> &'static str {
        match self {
            Tool::Cargo => "cargo",
            Tool::Rustc => "rustc",
            Tool::Rustup => "rustup",
            Tool::Rustfmt => "rustfmt",
        }
    }
}

fn invoke(list: &[fn(&str) -> Option<PathBuf>], executable: &str) -> PathBuf {
    list.iter().find_map(|it| it(executable)).unwrap_or_else(|| executable.into())
}

/// Looks up the binary as its SCREAMING upper case in the env variables.
fn lookup_as_env_var(executable_name: &str) -> Option<PathBuf> {
    env::var_os(executable_name.to_ascii_uppercase()).map(Into::into)
}

/// Looks up the binary in the cargo home directory if it exists.
fn cargo_proxy(executable_name: &str) -> Option<PathBuf> {
    let mut path = get_cargo_home()?;
    path.push("bin");
    path.push(executable_name);
    probe_for_binary(path)
}

fn get_cargo_home() -> Option<PathBuf> {
    if let Some(path) = env::var_os("CARGO_HOME") {
        return Some(path.into());
    }

    if let Some(mut path) = home::home_dir() {
        path.push(".cargo");
        return Some(path);
    }

    None
}

fn lookup_in_path(exec: &str) -> Option<PathBuf> {
    let paths = env::var_os("PATH").unwrap_or_default();
    env::split_paths(&paths).map(|path| path.join(exec)).find_map(probe_for_binary)
}

pub fn probe_for_binary(path: PathBuf) -> Option<PathBuf> {
    let with_extension = match env::consts::EXE_EXTENSION {
        "" => None,
        it => Some(path.with_extension(it)),
    };
    iter::once(path).chain(with_extension).find(|it| it.is_file())
}
