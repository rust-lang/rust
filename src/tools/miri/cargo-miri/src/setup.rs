//! Implements `cargo miri setup`.

use std::env;
use std::ffi::OsStr;
use std::path::PathBuf;
use std::process::{self, Command};

use rustc_build_sysroot::{BuildMode, Sysroot, SysrootConfig};
use rustc_version::VersionMeta;

use crate::util::*;

/// Performs the setup required to make `cargo miri` work: Getting a custom-built libstd. Then sets
/// `MIRI_SYSROOT`. Skipped if `MIRI_SYSROOT` is already set, in which case we expect the user has
/// done all this already.
pub fn setup(subcommand: &MiriCommand, target: &str, rustc_version: &VersionMeta) {
    let only_setup = matches!(subcommand, MiriCommand::Setup);
    let ask_user = !only_setup;
    let print_sysroot = only_setup && has_arg_flag("--print-sysroot"); // whether we just print the sysroot path
    if std::env::var_os("MIRI_SYSROOT").is_some() {
        if only_setup {
            println!("WARNING: MIRI_SYSROOT already set, not doing anything.")
        }
        return;
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
    let user_dirs = directories::ProjectDirs::from("org", "rust-lang", "miri").unwrap();
    let sysroot_dir = user_dirs.cache_dir();
    // Sysroot configuration and build details.
    let sysroot_config = if std::env::var_os("MIRI_NO_STD").is_some() {
        SysrootConfig::NoStd
    } else {
        SysrootConfig::WithStd { std_features: &["panic_unwind", "backtrace"] }
    };
    let cargo_cmd = || {
        let mut command = cargo();
        // Use Miri as rustc to build a libstd compatible with us (and use the right flags).
        // However, when we are running in bootstrap, we cannot just overwrite `RUSTC`,
        // because we still need bootstrap to distinguish between host and target crates.
        // In that case we overwrite `RUSTC_REAL` instead which determines the rustc used
        // for target crates.
        // We set ourselves (`cargo-miri`) instead of Miri directly to be able to patch the flags
        // for `libpanic_abort` (usually this is done by bootstrap but we have to do it ourselves).
        // The `MIRI_CALLED_FROM_SETUP` will mean we dispatch to `phase_setup_rustc`.
        let cargo_miri_path = std::env::current_exe().expect("current executable path invalid");
        if env::var_os("RUSTC_STAGE").is_some() {
            assert!(env::var_os("RUSTC").is_some());
            command.env("RUSTC_REAL", &cargo_miri_path);
        } else {
            command.env("RUSTC", &cargo_miri_path);
        }
        command.env("MIRI_CALLED_FROM_SETUP", "1");
        // Make sure there are no other wrappers getting in our way (Cc
        // https://github.com/rust-lang/miri/issues/1421,
        // https://github.com/rust-lang/miri/issues/2429). Looks like setting
        // `RUSTC_WRAPPER` to the empty string overwrites `build.rustc-wrapper` set via
        // `config.toml`.
        command.env("RUSTC_WRAPPER", "");

        if only_setup {
            if print_sysroot {
                // Be extra sure there is no noise on stdout.
                command.stdout(process::Stdio::null());
            }
        } else {
            command.stdout(process::Stdio::null());
            command.stderr(process::Stdio::null());
        }
        // Disable debug assertions in the standard library -- Miri is already slow enough.
        // But keep the overflow checks, they are cheap. This completely overwrites flags
        // the user might have set, which is consistent with normal `cargo build` that does
        // not apply `RUSTFLAGS` to the sysroot either.
        let rustflags = vec!["-Cdebug-assertions=off".into(), "-Coverflow-checks=on".into()];
        (command, rustflags)
    };
    // Make sure all target-level Miri invocations know their sysroot.
    std::env::set_var("MIRI_SYSROOT", sysroot_dir);

    // Do the build.
    if only_setup {
        // We want to be explicit.
        eprintln!("Preparing a sysroot for Miri (target: {target})...");
    } else {
        // We want to be quiet, but still let the user know that something is happening.
        eprint!("Preparing a sysroot for Miri (target: {target})... ");
    }
    Sysroot::new(sysroot_dir, target)
        .build_from_source(&rust_src, BuildMode::Check, sysroot_config, rustc_version, cargo_cmd)
        .unwrap_or_else(|_| {
            if only_setup {
                show_error!("failed to build sysroot, see error details above")
            } else {
                show_error!(
                    "failed to build sysroot; run `cargo miri setup` to see the error details"
                )
            }
        });
    if only_setup {
        eprintln!("A sysroot for Miri is now available in `{}`.", sysroot_dir.display());
    } else {
        eprintln!("done");
    }
    if print_sysroot {
        // Print just the sysroot and nothing else to stdout; this way we do not need any escaping.
        println!("{}", sysroot_dir.display());
    }
}
