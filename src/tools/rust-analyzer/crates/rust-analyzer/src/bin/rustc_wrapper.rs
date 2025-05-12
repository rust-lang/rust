//! We setup RUSTC_WRAPPER to point to `rust-analyzer` binary itself during the
//! initial `cargo check`. That way, we avoid checking the actual project, and
//! only build proc macros and build.rs.
//!
//! Code taken from IntelliJ :0)
//!     https://github.com/intellij-rust/intellij-rust/blob/master/native-helper/src/main.rs
use std::{
    ffi::OsString,
    io,
    process::{Command, ExitCode, Stdio},
};

pub(crate) fn main() -> io::Result<ExitCode> {
    let mut args = std::env::args_os();
    let _me = args.next().unwrap();
    let rustc = args.next().unwrap();
    run_rustc_skipping_cargo_checking(rustc, args.collect())
}

fn run_rustc_skipping_cargo_checking(
    rustc_executable: OsString,
    args: Vec<OsString>,
) -> io::Result<ExitCode> {
    // `CARGO_CFG_TARGET_ARCH` is only set by cargo when executing build scripts
    // We don't want to exit out checks unconditionally with success if a build
    // script tries to invoke checks themselves
    // See https://github.com/rust-lang/rust-analyzer/issues/12973 for context
    let not_invoked_by_build_script = std::env::var_os("CARGO_CFG_TARGET_ARCH").is_none();
    let is_cargo_check = args.iter().any(|arg| {
        let arg = arg.to_string_lossy();
        // `cargo check` invokes `rustc` with `--emit=metadata` argument.
        //
        // https://doc.rust-lang.org/rustc/command-line-arguments.html#--emit-specifies-the-types-of-output-files-to-generate
        // link —     Generates the crates specified by --crate-type. The default
        //            output filenames depend on the crate type and platform. This
        //            is the default if --emit is not specified.
        // metadata — Generates a file containing metadata about the crate.
        //            The default output filename is CRATE_NAME.rmeta.
        arg.starts_with("--emit=") && arg.contains("metadata") && !arg.contains("link")
    });
    if not_invoked_by_build_script && is_cargo_check {
        Ok(ExitCode::from(0))
    } else {
        run_rustc(rustc_executable, args)
    }
}

fn run_rustc(rustc_executable: OsString, args: Vec<OsString>) -> io::Result<ExitCode> {
    #[allow(clippy::disallowed_methods)]
    let mut child = Command::new(rustc_executable)
        .args(args)
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()?;
    Ok(ExitCode::from(child.wait()?.code().unwrap_or(102) as u8))
}
