//! Implements `cargo miri setup`.

use std::env;
use std::ffi::OsStr;
use std::path::PathBuf;
use std::process::{self, Command};

use rustc_build_sysroot::{BuildMode, SysrootBuilder, SysrootConfig, SysrootStatus};
use rustc_version::VersionMeta;

use crate::util::*;

/// Performs the setup required to make `cargo miri` work: Getting a custom-built libstd. Then sets
/// `MIRI_SYSROOT`. Skipped if `MIRI_SYSROOT` is already set, in which case we expect the user has
/// done all this already.
pub fn setup(
    subcommand: &MiriCommand,
    target: &str,
    rustc_version: &VersionMeta,
    verbose: usize,
    quiet: bool,
) -> PathBuf {
    let only_setup = matches!(subcommand, MiriCommand::Setup);
    let ask_user = !only_setup;
    let print_sysroot = only_setup && has_arg_flag("--print-sysroot"); // whether we just print the sysroot path
    let show_setup = only_setup && !print_sysroot;
    if !only_setup {
        if let Some(sysroot) = std::env::var_os("MIRI_SYSROOT") {
            // Skip setup step if MIRI_SYSROOT is explicitly set, *unless* we are `cargo miri setup`.
            return sysroot.into();
        }
    }

    // Determine where the rust sources are located.  The env var trumps auto-detection.
    let rust_src_env_var = std::env::var_os("MIRI_LIB_SRC");
    let rust_src = match rust_src_env_var {
        Some(path) => {
            let path = PathBuf::from(path);
            // Make path absolute if possible.
            path.canonicalize().unwrap_or(path)
        }
        None => {
            // Check for `rust-src` rustup component.
            let rustup_src = rustc_build_sysroot::rustc_sysroot_src(miri_for_host())
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
        show_error!("given Rust source directory `{}` does not exist.", rust_src.display());
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
    let no_std = match std::env::var_os("MIRI_NO_STD") {
        None =>
        // No-std heuristic taken from rust/src/bootstrap/config.rs
        // (https://github.com/rust-lang/rust/blob/25b5af1b3a0b9e2c0c57b223b2d0e3e203869b2c/src/bootstrap/config.rs#L549-L555).
            target.contains("-none")
                || target.contains("nvptx")
                || target.contains("switch")
                || target.contains("-uefi"),
        Some(val) => val != "0",
    };
    let sysroot_config = if no_std {
        SysrootConfig::NoStd
    } else {
        SysrootConfig::WithStd {
            std_features: ["panic_unwind", "backtrace"].into_iter().map(Into::into).collect(),
        }
    };
    let cargo_cmd = {
        let mut command = cargo();
        // Use Miri as rustc to build a libstd compatible with us (and use the right flags).
        // We set ourselves (`cargo-miri`) instead of Miri directly to be able to patch the flags
        // for `libpanic_abort` (usually this is done by bootstrap but we have to do it ourselves).
        // The `MIRI_CALLED_FROM_SETUP` will mean we dispatch to `phase_setup_rustc`.
        // However, when we are running in bootstrap, we cannot just overwrite `RUSTC`,
        // because we still need bootstrap to distinguish between host and target crates.
        // In that case we overwrite `RUSTC_REAL` instead which determines the rustc used
        // for target crates.
        let cargo_miri_path = std::env::current_exe().expect("current executable path invalid");
        if env::var_os("RUSTC_STAGE").is_some() {
            assert!(
                env::var_os("RUSTC").is_some() && env::var_os("RUSTC_WRAPPER").is_some(),
                "cargo-miri setup is running inside rustc bootstrap but RUSTC or RUST_WRAPPER is not set"
            );
            command.env("RUSTC_REAL", &cargo_miri_path);
        } else {
            command.env("RUSTC", &cargo_miri_path);
        }
        command.env("MIRI_CALLED_FROM_SETUP", "1");
        // Miri expects `MIRI_SYSROOT` to be set when invoked in target mode. Even if that directory is empty.
        command.env("MIRI_SYSROOT", &sysroot_dir);
        // Make sure there are no other wrappers getting in our way (Cc
        // https://github.com/rust-lang/miri/issues/1421,
        // https://github.com/rust-lang/miri/issues/2429). Looks like setting
        // `RUSTC_WRAPPER` to the empty string overwrites `build.rustc-wrapper` set via
        // `bootstrap.toml`.
        command.env("RUSTC_WRAPPER", "");

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
    // Disable debug assertions in the standard library -- Miri is already slow enough.
    // But keep the overflow checks, they are cheap. This completely overwrites flags
    // the user might have set, which is consistent with normal `cargo build` that does
    // not apply `RUSTFLAGS` to the sysroot either.
    let rustflags = &["-Cdebug-assertions=off", "-Coverflow-checks=on"];

    let mut after_build_output = String::new(); // what should be printed when the build is done.
    let notify = || {
        if !quiet {
            eprint!("Preparing a sysroot for Miri (target: {target})");
            if verbose > 0 {
                eprint!(" in {}", sysroot_dir.display());
            }
            if show_setup {
                // Cargo will print things, so we need to finish this line.
                eprintln!("...");
                after_build_output = format!(
                    "A sysroot for Miri is now available in `{}`.\n",
                    sysroot_dir.display()
                );
            } else {
                // Keep all output on a single line.
                eprint!("... ");
                after_build_output = format!("done\n");
            }
        }
    };

    // Do the build.
    let status = SysrootBuilder::new(&sysroot_dir, target)
        .build_mode(BuildMode::Check)
        .rustc_version(rustc_version.clone())
        .sysroot_config(sysroot_config)
        .rustflags(rustflags)
        .cargo(cargo_cmd)
        .when_build_required(notify)
        .build_from_source(&rust_src);
    match status {
        Ok(SysrootStatus::AlreadyCached) =>
            if !quiet && show_setup {
                eprintln!(
                    "A sysroot for Miri is already available in `{}`.",
                    sysroot_dir.display()
                );
            },
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
