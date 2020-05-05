//! This crate contains a single public function
//! [`get_path_for_executable`](fn.get_path_for_executable.html).
//! See docs there for more information.

use anyhow::{Error, Result};
use std::env;
use std::path::Path;
use std::process::Command;

/// Return a `String` to use for the given executable.
///
/// E.g., `get_path_for_executable("cargo")` may return just `cargo` if that
/// gives a valid Cargo executable; or it may return a full path to a valid
/// Cargo.
pub fn get_path_for_executable(executable_name: impl AsRef<str>) -> Result<String> {
    // The current implementation checks three places for an executable to use:
    // 1) Appropriate environment variable (erroring if this is set but not a usable executable)
    //      example: for cargo, this checks $CARGO environment variable; for rustc, $RUSTC; etc
    // 2) `<executable_name>`
    //      example: for cargo, this tries just `cargo`, which will succeed if `cargo` is on the $PATH
    // 3) `~/.cargo/bin/<executable_name>`
    //      example: for cargo, this tries ~/.cargo/bin/cargo
    //      It seems that this is a reasonable place to try for cargo, rustc, and rustup
    let executable_name = executable_name.as_ref();
    let env_var = executable_name.to_ascii_uppercase();
    if let Ok(path) = env::var(&env_var) {
        if is_valid_executable(&path) {
            Ok(path)
        } else {
            Err(Error::msg(format!(
                "`{}` environment variable points to something that's not a valid executable",
                env_var
            )))
        }
    } else {
        let final_path: Option<String> = if is_valid_executable(executable_name) {
            Some(executable_name.to_owned())
        } else {
            if let Some(mut path) = dirs::home_dir() {
                path.push(".cargo");
                path.push("bin");
                path.push(executable_name);
                if is_valid_executable(&path) {
                    Some(path.into_os_string().into_string().expect("Invalid Unicode in path"))
                } else {
                    None
                }
            } else {
                None
            }
        };
        final_path.ok_or(
            // This error message may also be caused by $PATH or $CARGO/$RUSTC/etc not being set correctly
            // for VSCode, even if they are set correctly in a terminal.
            // On macOS in particular, launching VSCode from terminal with `code <dirname>` causes VSCode
            // to inherit environment variables including $PATH, $CARGO, $RUSTC, etc from that terminal;
            // but launching VSCode from Dock does not inherit environment variables from a terminal.
            // For more discussion, see #3118.
            Error::msg(format!("Failed to find `{}` executable. Make sure `{}` is in `$PATH`, or set `${}` to point to a valid executable.", executable_name, executable_name, env_var))
        )
    }
}

/// Does the given `Path` point to a usable executable?
///
/// (assumes the executable takes a `--version` switch and writes to stdout,
/// which is true for `cargo`, `rustc`, and `rustup`)
fn is_valid_executable(p: impl AsRef<Path>) -> bool {
    Command::new(p.as_ref()).arg("--version").output().is_ok()
}
