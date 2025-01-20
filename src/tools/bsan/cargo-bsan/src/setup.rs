use std::ffi::OsStr;
use std::path::PathBuf;
use std::process;
use std::process::Command;

use rustc_build_sysroot::BuildMode;
use rustc_build_sysroot::SysrootBuilder;
use rustc_build_sysroot::SysrootConfig;
use rustc_build_sysroot::SysrootStatus;
use rustc_version::VersionMeta;

use crate::BSANCommand;
use crate::arg::*;
use crate::util::*;

pub fn setup(
    subcommand: &BSANCommand,
    target: &str,
    rustc_version: &VersionMeta,
    verbose: usize,
    quiet: bool,
) -> PathBuf {
    let only_setup = matches!(subcommand, BSANCommand::Setup);
    let ask_user = !only_setup;
    let print_sysroot = only_setup && has_arg_flag("--print-sysroot"); // whether we just print the sysroot path
    let show_setup = only_setup && !print_sysroot;
    if !only_setup {
        if let Some(sysroot) = std::env::var_os("BSAN_SYSROOT") {
            // Skip setup step if BSAN_SYSROOT is explicitly set, *unless* we are `cargo bsan setup`.
            return sysroot.into();
        }
    }

    // Determine where the rust sources are located.  
    // The env var trumps auto-detection.
    let rust_src_env_var = std::env::var_os("BSAN_LIB_SRC");
    let rust_src = match rust_src_env_var {
        Some(path) => {
            let path = PathBuf::from(path);
            // Make path absolute if possible.
            path.canonicalize().unwrap_or(path)
        }
        None => {
            // Check for `rust-src` rustup component.
            let rustup_src = rustc_build_sysroot::rustc_sysroot_src(bsan_for_host())
                .expect("could not determine sysroot source directory");
            if !rustup_src.exists() {
                // Ask the user to install the `rust-src` component, and use that.
                let mut cmd = Command::new("rustup");
                cmd.args(["component", "add", "rust-src"]);
                ask_to_run(
                    cmd,
                    ask_user,
                    "install the `rust-src` component for the selected toolchain",
                );
            }
            rustup_src
        }
    };
    if !rust_src.exists() {
        show_error!(
            "given Rust source directory `{}` does not exist.",
            rust_src.display()
        );
    }
    if rust_src.file_name().and_then(OsStr::to_str) != Some("library") {
        show_error!(
            "given Rust source directory `{}` does not seem to be the `library` subdirectory of \
             a Rust source checkout.",
            rust_src.display()
        );
    }

    // Determine where to put the sysroot.
    let sysroot_dir = get_sysroot_dir();

    // Sysroot configuration and build details.
    let no_std = match std::env::var_os("BSAN_NO_STD") {
        None =>
        // No-std heuristic taken from rust/src/bootstrap/config.rs
        // (https://github.com/rust-lang/rust/blob/25b5af1b3a0b9e2c0c57b223b2d0e3e203869b2c/src/bootstrap/config.rs#L549-L555).
        {
            target.contains("-none")
                || target.contains("nvptx")
                || target.contains("switch")
                || target.contains("-uefi")
        }
        Some(val) => val != "0",
    };
    let sysroot_config = if no_std {
        SysrootConfig::NoStd
    } else {
        SysrootConfig::WithStd {
            std_features: ["panic_unwind", "backtrace"]
                .into_iter()
                .map(Into::into)
                .collect(),
        }
    };
    let cargo_cmd = {
        let mut command = cargo();
        command.env("BSAN_SYSROOT", &sysroot_dir);

        // Ensure that the standard library is also instrumented.
        command.env("RUSTC_WRAPPER", find_bsan());

        if show_setup {
            // Forward output. Even make it verbose, if requested.
            command.stdout(process::Stdio::inherit());
            command.stderr(process::Stdio::inherit());
            for _ in 0..verbose {
                command.arg("-v");
            }
            if quiet {
                command.arg("--quiet");
            }
        }
        command
    };
    // Disable debug assertions in the standard library
    // But keep the overflow checks, they are cheap. This completely overwrites flags
    // the user might have set, which is consistent with normal `cargo build` that does
    // not apply `RUSTFLAGS` to the sysroot either.
    let rustflags = &["-Cdebug-assertions=off", "-Coverflow-checks=on"];

    let mut after_build_output = String::new(); // what should be printed when the build is done.
    let notify = || {
        if !quiet {
            eprint!("Preparing a sysroot for BorrowSanitizer (target: {target})");
            if verbose > 0 {
                eprint!(" in {}", sysroot_dir.display());
            }
            if show_setup {
                // Cargo will print things, so we need to finish this line.
                eprintln!("...");
                after_build_output = format!(
                    "A sysroot for BorrowSanitizer is now available in `{}`.\n",
                    sysroot_dir.display()
                );
            } else {
                // Keep all output on a single line.
                eprint!("... ");
                after_build_output = "done\n".to_string();
            }
        }
    };

    // Do the build.
    let status = SysrootBuilder::new(&sysroot_dir, target)
        .build_mode(BuildMode::Build)
        .rustc_version(rustc_version.clone())
        .sysroot_config(sysroot_config)
        .rustflags(rustflags)
        .cargo(cargo_cmd)
        .when_build_required(notify)
        .build_from_source(&rust_src);

    match status {
        Ok(SysrootStatus::AlreadyCached) => {
            if !quiet && show_setup {
                eprintln!(
                    "A sysroot for BorrowSanitizer is already available in `{}`.",
                    sysroot_dir.display()
                );
            }
        }
        Ok(SysrootStatus::SysrootBuilt) => {
            // Print what `notify` prepared.
            eprint!("{after_build_output}");
        }
        Err(err) => show_error!("failed to build sysroot: {err:?}"),
    }

    if print_sysroot {
        // Print just the sysroot and nothing else to stdout; this way we do not need any escaping.
        println!("{}", sysroot_dir.display());
    }
    sysroot_dir
}
