//! This module serves two purposes:
//!     1. It is part of the `utils` module and used in other parts of bootstrap.
//!     2. It is embedded inside bootstrap shims to avoid a dependency on the bootstrap library.
//!        Therefore, this module should never use any other bootstrap module. This reduces binary
//!        size and improves compilation time by minimizing linking time.

#![allow(dead_code)]

use std::env;
use std::ffi::OsString;
use std::fs::OpenOptions;
use std::io::Write;
use std::process::Command;
use std::str::FromStr;

// If we were to declare a tests submodule here, the shim binaries that include this
// module via `#[path]` would fail to find it, which breaks `./x check bootstrap`.
// So instead the unit tests for this module are in `super::tests::shared_helpers_tests`.

/// Returns the environment variable which the dynamic library lookup path
/// resides in for this platform.
pub fn dylib_path_var() -> &'static str {
    if cfg!(target_os = "windows") {
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
    if target.contains("windows") {
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
///
/// If "RUSTC_STAGE" was not set, the program will be terminated with 101.
pub fn parse_rustc_stage() -> String {
    env::var("RUSTC_STAGE").unwrap_or_else(|_| {
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
