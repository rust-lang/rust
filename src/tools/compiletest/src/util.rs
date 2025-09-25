use std::env;
use std::process::Command;

use camino::{Utf8Path, Utf8PathBuf};

#[cfg(test)]
mod tests;

pub fn make_new_path(path: &str) -> String {
    assert!(cfg!(windows));
    // Windows just uses PATH as the library search path, so we have to
    // maintain the current value while adding our own
    match env::var(lib_path_env_var()) {
        Ok(curr) => format!("{}{}{}", path, path_div(), curr),
        Err(..) => path.to_owned(),
    }
}

pub fn lib_path_env_var() -> &'static str {
    "PATH"
}
fn path_div() -> &'static str {
    ";"
}

pub trait Utf8PathBufExt {
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
pub fn dylib_env_var() -> &'static str {
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
pub fn add_dylib_path(
    cmd: &mut Command,
    paths: impl Iterator<Item = impl Into<std::path::PathBuf>>,
) {
    let path_env = env::var_os(dylib_env_var());
    let old_paths = path_env.as_ref().map(env::split_paths);
    let new_paths = paths.map(Into::into).chain(old_paths.into_iter().flatten());
    cmd.env(dylib_env_var(), env::join_paths(new_paths).unwrap());
}

pub fn copy_dir_all(src: &Utf8Path, dst: &Utf8Path) -> std::io::Result<()> {
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
    ($(#[$meta:meta])* $vis:vis enum $name:ident { $($variant:ident => $repr:expr,)* }) => {
        $(#[$meta])*
        $vis enum $name {
            $($variant,)*
        }

        impl $name {
            $vis const VARIANTS: &'static [Self] = &[$(Self::$variant,)*];
            $vis const STR_VARIANTS: &'static [&'static str] = &[$(Self::$variant.to_str(),)*];

            $vis const fn to_str(&self) -> &'static str {
                match self {
                    $(Self::$variant => $repr,)*
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
                    $($repr => Ok(Self::$variant),)*
                    _ => Err(format!(concat!("unknown `", stringify!($name), "` variant: `{}`"), s)),
                }
            }
        }
    }
}

pub(crate) use string_enum;
