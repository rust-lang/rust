pub mod cc;
pub mod run;
pub mod rustc;
pub mod rustdoc;

use std::env;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

pub use object;
pub use wasmparser;

pub use cc::{cc, extra_c_flags, extra_cxx_flags, Cc};
pub use run::{run, run_fail};
pub use rustc::{aux_build, rustc, Rustc};
pub use rustdoc::{bare_rustdoc, rustdoc, Rustdoc};

/// Path of `TMPDIR` (a temporary build directory, not under `/tmp`).
pub fn tmp_dir() -> PathBuf {
    env::var_os("TMPDIR").unwrap().into()
}

/// `TARGET`
pub fn target() -> String {
    env::var("TARGET").unwrap()
}

/// Check if target is windows-like.
pub fn is_windows() -> bool {
    env::var_os("IS_WINDOWS").is_some()
}

/// Check if target uses msvc.
pub fn is_msvc() -> bool {
    env::var_os("IS_MSVC").is_some()
}

/// Construct a path to a static library under `$TMPDIR` given the library name. This will return a
/// path with `$TMPDIR` joined with platform-and-compiler-specific library name.
pub fn static_lib(name: &str) -> PathBuf {
    tmp_dir().join(static_lib_name(name))
}

/// Construct the static library name based on the platform.
pub fn static_lib_name(name: &str) -> String {
    // See tools.mk (irrelevant lines omitted):
    //
    // ```makefile
    // ifeq ($(UNAME),Darwin)
    //     STATICLIB = $(TMPDIR)/lib$(1).a
    // else
    //     ifdef IS_WINDOWS
    //         ifdef IS_MSVC
    //             STATICLIB = $(TMPDIR)/$(1).lib
    //         else
    //             STATICLIB = $(TMPDIR)/lib$(1).a
    //         endif
    //     else
    //         STATICLIB = $(TMPDIR)/lib$(1).a
    //     endif
    // endif
    // ```
    assert!(!name.contains(char::is_whitespace), "name cannot contain whitespace");

    if target().contains("msvc") { format!("{name}.lib") } else { format!("lib{name}.a") }
}

/// Construct the binary name based on platform.
pub fn bin_name(name: &str) -> String {
    if is_windows() { format!("{name}.exe") } else { name.to_string() }
}

/// Use `cygpath -w` on a path to get a Windows path string back. This assumes that `cygpath` is
/// available on the platform!
#[track_caller]
pub fn cygpath_windows<P: AsRef<Path>>(path: P) -> String {
    let caller_location = std::panic::Location::caller();
    let caller_line_number = caller_location.line();

    let mut cygpath = Command::new("cygpath");
    cygpath.arg("-w");
    cygpath.arg(path.as_ref());
    let output = cygpath.output().unwrap();
    if !output.status.success() {
        handle_failed_output(&format!("{:#?}", cygpath), output, caller_line_number);
    }
    let s = String::from_utf8(output.stdout).unwrap();
    // cygpath -w can attach a newline
    s.trim().to_string()
}

/// Run `uname`. This assumes that `uname` is available on the platform!
#[track_caller]
pub fn uname() -> String {
    let caller_location = std::panic::Location::caller();
    let caller_line_number = caller_location.line();

    let mut uname = Command::new("uname");
    let output = uname.output().unwrap();
    if !output.status.success() {
        handle_failed_output(&format!("{:#?}", uname), output, caller_line_number);
    }
    String::from_utf8(output.stdout).unwrap()
}

fn handle_failed_output(cmd: &str, output: Output, caller_line_number: u32) -> ! {
    if output.status.success() {
        eprintln!("command incorrectly succeeded at line {caller_line_number}");
    } else {
        eprintln!("command failed at line {caller_line_number}");
    }
    eprintln!("{cmd}");
    eprintln!("output status: `{}`", output.status);
    eprintln!("=== STDOUT ===\n{}\n\n", String::from_utf8(output.stdout).unwrap());
    eprintln!("=== STDERR ===\n{}\n\n", String::from_utf8(output.stderr).unwrap());
    std::process::exit(1)
}
