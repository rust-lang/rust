//! This file is meant to be included directly from bootstrap shims to avoid a
//! dependency on the bootstrap library. This reduces the binary size and
//! improves compilation time by reducing the linking time.

use std::env;
use std::fs::OpenOptions;
use std::io::Write;
use std::process::Command;
use std::str::FromStr;

/// Parses the value of the "RUSTC_VERBOSE" environment variable and returns it as a `usize`.
/// If it was not defined, returns 0 by default.
///
/// Panics if "RUSTC_VERBOSE" is defined with the value that is not an unsigned integer.
pub(crate) fn parse_rustc_verbose() -> usize {
    match env::var("RUSTC_VERBOSE") {
        Ok(s) => usize::from_str(&s).expect("RUSTC_VERBOSE should be an integer"),
        Err(_) => 0,
    }
}

/// Parses the value of the "RUSTC_STAGE" environment variable and returns it as a `String`.
///
/// If "RUSTC_STAGE" was not set, the program will be terminated with 101.
pub(crate) fn parse_rustc_stage() -> String {
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
pub(crate) fn maybe_dump(dump_name: String, cmd: &Command) {
    if let Ok(dump_dir) = env::var("DUMP_BOOTSTRAP_SHIMS") {
        let dump_file = format!("{dump_dir}/{dump_name}");

        let mut file = OpenOptions::new().create(true).append(true).open(dump_file).unwrap();

        let cmd_dump = format!("{:?}\n", cmd);
        let cmd_dump = cmd_dump.replace(&env::var("BUILD_OUT").unwrap(), "${BUILD_OUT}");
        let cmd_dump = cmd_dump.replace(&env::var("CARGO_HOME").unwrap(), "${CARGO_HOME}");

        file.write_all(cmd_dump.as_bytes()).expect("Unable to write file");
    }
}
