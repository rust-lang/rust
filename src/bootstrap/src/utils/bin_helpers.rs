//! This file is meant to be included directly from bootstrap shims to avoid a
//! dependency on the bootstrap library. This reduces the binary size and
//! improves compilation time by reducing the linking time.

/// Parses the value of the "RUSTC_VERBOSE" environment variable and returns it as a `usize`.
/// If it was not defined, returns 0 by default.
///
/// Panics if "RUSTC_VERBOSE" is defined with the value that is not an unsigned integer.
pub(crate) fn parse_rustc_verbose() -> usize {
    use std::str::FromStr;

    match std::env::var("RUSTC_VERBOSE") {
        Ok(s) => usize::from_str(&s).expect("RUSTC_VERBOSE should be an integer"),
        Err(_) => 0,
    }
}

/// Parses the value of the "RUSTC_STAGE" environment variable and returns it as a `String`.
///
/// If "RUSTC_STAGE" was not set, the program will be terminated with 101.
#[allow(unused)]
pub(crate) fn parse_rustc_stage() -> String {
    std::env::var("RUSTC_STAGE").unwrap_or_else(|_| {
        // Don't panic here; it's reasonable to try and run these shims directly. Give a helpful error instead.
        eprintln!("rustc shim: fatal: RUSTC_STAGE was not set");
        eprintln!("rustc shim: note: use `x.py build -vvv` to see all environment variables set by bootstrap");
        std::process::exit(101);
    })
}
