pub mod run;
pub mod rustc;
pub mod rustdoc;

use std::env;
use std::path::PathBuf;
use std::process::Output;

pub use object;
pub use wasmparser;

pub use run::{run, run_fail};
pub use rustc::{aux_build, rustc, Rustc};
pub use rustdoc::{bare_rustdoc, rustdoc, Rustdoc};

/// Path of `TMPDIR` (a temporary build directory, not under `/tmp`).
pub fn tmp_dir() -> PathBuf {
    env::var_os("TMPDIR").unwrap().into()
}

fn handle_failed_output(cmd: &str, output: Output, caller_line_number: u32) -> ! {
    eprintln!("command failed at line {caller_line_number}");
    eprintln!("{cmd}");
    eprintln!("output status: `{}`", output.status);
    eprintln!("=== STDOUT ===\n{}\n\n", String::from_utf8(output.stdout).unwrap());
    eprintln!("=== STDERR ===\n{}\n\n", String::from_utf8(output.stderr).unwrap());
    std::process::exit(1)
}
