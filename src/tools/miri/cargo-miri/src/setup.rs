//! Implements `cargo miri setup` via xargo

use std::env;
use std::ffi::OsStr;
use std::fs::{self};
use std::io::BufRead;
use std::ops::Not;
use std::path::{Path, PathBuf};
use std::process::{self, Command};

use crate::{util::*, version::*};

fn xargo_version() -> Option<(u32, u32, u32)> {
    let out = xargo_check().arg("--version").output().ok()?;
    if !out.status.success() {
        return None;
    }
    // Parse output. The first line looks like "xargo 0.3.12 (b004f1c 2018-12-13)".
    let line = out
        .stderr
        .lines()
        .next()
        .expect("malformed `xargo --version` output: not at least one line")
        .expect("malformed `xargo --version` output: error reading first line");
    let (name, version) = {
        let mut split = line.split(' ');
        (
            split.next().expect("malformed `xargo --version` output: empty"),
            split.next().expect("malformed `xargo --version` output: not at least two words"),
        )
    };
    if name != "xargo" {
        // This is some fork of xargo
        return None;
    }
    let mut version_pieces = version.split('.');
    let major = version_pieces
        .next()
        .expect("malformed `xargo --version` output: not a major version piece")
        .parse()
        .expect("malformed `xargo --version` output: major version is not an integer");
    let minor = version_pieces
        .next()
        .expect("malformed `xargo --version` output: not a minor version piece")
        .parse()
        .expect("malformed `xargo --version` output: minor version is not an integer");
    let patch = version_pieces
        .next()
        .expect("malformed `xargo --version` output: not a patch version piece")
        .parse()
        .expect("malformed `xargo --version` output: patch version is not an integer");
    if version_pieces.next().is_some() {
        panic!("malformed `xargo --version` output: more than three pieces in version");
    }
    Some((major, minor, patch))
}

/// Performs the setup required to make `cargo miri` work: Getting a custom-built libstd. Then sets
/// `MIRI_SYSROOT`. Skipped if `MIRI_SYSROOT` is already set, in which case we expect the user has
/// done all this already.
pub fn setup(subcommand: &MiriCommand, host: &str, target: &str) {
    let only_setup = matches!(subcommand, MiriCommand::Setup);
    let ask_user = !only_setup;
    let print_sysroot = only_setup && has_arg_flag("--print-sysroot"); // whether we just print the sysroot path
    if std::env::var_os("MIRI_SYSROOT").is_some() {
        if only_setup {
            println!("WARNING: MIRI_SYSROOT already set, not doing anything.")
        }
        return;
    }

    // First, we need xargo.
    if xargo_version().map_or(true, |v| v < XARGO_MIN_VERSION) {
        if std::env::var_os("XARGO_CHECK").is_some() {
            // The user manually gave us a xargo binary; don't do anything automatically.
            show_error!("xargo is too old; please upgrade to the latest version")
        }
        let mut cmd = cargo();
        cmd.args(["install", "xargo"]);
        ask_to_run(cmd, ask_user, "install a recent enough xargo");
    }

    // Determine where the rust sources are located.  The env vars manually setting the source
    // (`MIRI_LIB_SRC`, `XARGO_RUST_SRC`) trump auto-detection.
    let rust_src_env_var =
        std::env::var_os("MIRI_LIB_SRC").or_else(|| std::env::var_os("XARGO_RUST_SRC"));
    let rust_src = match rust_src_env_var {
        Some(path) => {
            let path = PathBuf::from(path);
            // Make path absolute if possible.
            path.canonicalize().unwrap_or(path)
        }
        None => {
            // Check for `rust-src` rustup component.
            let output = miri_for_host()
                .args(["--print", "sysroot"])
                .output()
                .expect("failed to determine sysroot");
            if !output.status.success() {
                show_error!(
                    "Failed to determine sysroot; Miri said:\n{}",
                    String::from_utf8_lossy(&output.stderr).trim_end()
                );
            }
            let sysroot = std::str::from_utf8(&output.stdout).unwrap();
            let sysroot = Path::new(sysroot.trim_end_matches('\n'));
            // Check for `$SYSROOT/lib/rustlib/src/rust/library`; test if that contains `std/Cargo.toml`.
            let rustup_src =
                sysroot.join("lib").join("rustlib").join("src").join("rust").join("library");
            if !rustup_src.join("std").join("Cargo.toml").exists() {
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

    // Next, we need our own libstd. Prepare a xargo project for that purpose.
    // We will do this work in whatever is a good cache dir for this platform.
    let dirs = directories::ProjectDirs::from("org", "rust-lang", "miri").unwrap();
    let dir = dirs.cache_dir();
    if !dir.exists() {
        fs::create_dir_all(dir).unwrap();
    }
    // The interesting bit: Xargo.toml (only needs content if we actually need std)
    let xargo_toml = if std::env::var_os("MIRI_NO_STD").is_some() {
        ""
    } else {
        r#"
[dependencies.std]
default_features = false
# We support unwinding, so enable that panic runtime.
features = ["panic_unwind", "backtrace"]

[dependencies.test]
"#
    };
    write_to_file(&dir.join("Xargo.toml"), xargo_toml);
    // The boring bits: a dummy project for xargo.
    // FIXME: With xargo-check, can we avoid doing this?
    write_to_file(
        &dir.join("Cargo.toml"),
        r#"
[package]
name = "miri-xargo"
description = "A dummy project for building libstd with xargo."
version = "0.0.0"

[lib]
path = "lib.rs"
"#,
    );
    write_to_file(&dir.join("lib.rs"), "#![no_std]");

    // Figure out where xargo will build its stuff.
    // Unfortunately, it puts things into a different directory when the
    // architecture matches the host.
    let sysroot = if target == host { dir.join("HOST") } else { PathBuf::from(dir) };
    // Make sure all target-level Miri invocations know their sysroot.
    std::env::set_var("MIRI_SYSROOT", &sysroot);

    // Now invoke xargo.
    let mut command = xargo_check();
    command.arg("check").arg("-q");
    command.current_dir(dir);
    command.env("XARGO_HOME", dir);
    command.env("XARGO_RUST_SRC", &rust_src);
    // We always need to set a target so rustc bootstrap can tell apart host from target crates.
    command.arg("--target").arg(target);
    // Use Miri as rustc to build a libstd compatible with us (and use the right flags).
    // However, when we are running in bootstrap, we cannot just overwrite `RUSTC`,
    // because we still need bootstrap to distinguish between host and target crates.
    // In that case we overwrite `RUSTC_REAL` instead which determines the rustc used
    // for target crates.
    // We set ourselves (`cargo-miri`) instead of Miri directly to be able to patch the flags
    // for `libpanic_abort` (usually this is done by bootstrap but we have to do it ourselves).
    // The `MIRI_CALLED_FROM_XARGO` will mean we dispatch to `phase_setup_rustc`.
    let cargo_miri_path = std::env::current_exe().expect("current executable path invalid");
    if env::var_os("RUSTC_STAGE").is_some() {
        assert!(env::var_os("RUSTC").is_some());
        command.env("RUSTC_REAL", &cargo_miri_path);
    } else {
        command.env("RUSTC", &cargo_miri_path);
    }
    command.env("MIRI_CALLED_FROM_XARGO", "1");
    // Make sure there are no other wrappers getting in our way
    // (Cc https://github.com/rust-lang/miri/issues/1421, https://github.com/rust-lang/miri/issues/2429).
    // Looks like setting `RUSTC_WRAPPER` to the empty string overwrites `build.rustc-wrapper` set via `config.toml`.
    command.env("RUSTC_WRAPPER", "");
    // Disable debug assertions in the standard library -- Miri is already slow enough. But keep the
    // overflow checks, they are cheap. This completely overwrites flags the user might have set,
    // which is consistent with normal `cargo build` that does not apply `RUSTFLAGS` to the sysroot
    // either.
    command.env("RUSTFLAGS", "-Cdebug-assertions=off -Coverflow-checks=on");
    // Manage the output the user sees.
    if only_setup {
        // We want to be explicit.
        eprintln!("Preparing a sysroot for Miri (target: {target})...");
        if print_sysroot {
            // Be extra sure there is no noise on stdout.
            command.stdout(process::Stdio::null());
        }
    } else {
        // We want to be quiet, but still let the user know that something is happening.
        eprint!("Preparing a sysroot for Miri (target: {target})... ");
        command.stdout(process::Stdio::null());
        command.stderr(process::Stdio::null());
    }

    // Finally run it!
    if command.status().expect("failed to run xargo").success().not() {
        if only_setup {
            show_error!("failed to run xargo, see error details above")
        } else {
            show_error!("failed to run xargo; run `cargo miri setup` to see the error details")
        }
    }

    // Figure out what to print.
    if only_setup {
        eprintln!("A sysroot for Miri is now available in `{}`.", sysroot.display());
    } else {
        eprintln!("done");
    }
    if print_sysroot {
        // Print just the sysroot and nothing else to stdout; this way we do not need any escaping.
        println!("{}", sysroot.display());
    }
}
