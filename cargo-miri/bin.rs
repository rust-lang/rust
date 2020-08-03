use std::env;
use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{self, BufRead, Write};
use std::ops::Not;
use std::path::{Path, PathBuf};
use std::process::Command;

use rustc_version::VersionMeta;

const XARGO_MIN_VERSION: (u32, u32, u32) = (0, 3, 22);

const CARGO_MIRI_HELP: &str = r#"Interprets bin crates and tests in Miri

Usage:
    cargo miri [subcommand] [<cargo options>...] [--] [<miri options>...] [--] [<program/test suite options>...]

Subcommands:
    run                      Run binaries (default)
    test                     Run tests
    setup                    Only perform automatic setup, but without asking questions (for getting a proper libstd)

Common options:
    -h, --help               Print this message
    --features               Features to compile for the package
    -V, --version            Print version info and exit

Other [options] are the same as `cargo check`.  Everything after the first "--" is
passed verbatim to Miri, which will pass everything after the second "--" verbatim
to the interpreted program.

Examples:
    cargo miri run -- -Zmiri-disable-stacked-borrows
    cargo miri test -- -- test-suite-filter
"#;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum MiriCommand {
    Run,
    Test,
    Setup,
}

fn show_help() {
    println!("{}", CARGO_MIRI_HELP);
}

fn show_version() {
    println!(
        "miri {} ({} {})",
        env!("CARGO_PKG_VERSION"),
        env!("VERGEN_SHA_SHORT"),
        env!("VERGEN_COMMIT_DATE")
    );
}

fn show_error(msg: String) -> ! {
    eprintln!("fatal error: {}", msg);
    std::process::exit(1)
}

// Determines whether a `--flag` is present.
fn has_arg_flag(name: &str) -> bool {
    let mut args = std::env::args().take_while(|val| val != "--");
    args.any(|val| val == name)
}

/// Gets the value of a `--flag`.
fn get_arg_flag_value(name: &str) -> Option<String> {
    // Stop searching at `--`.
    let mut args = std::env::args().take_while(|val| val != "--");
    loop {
        let arg = match args.next() {
            Some(arg) => arg,
            None => return None,
        };
        if !arg.starts_with(name) {
            continue;
        }
        // Strip leading `name`.
        let suffix = &arg[name.len()..];
        if suffix.is_empty() {
            // This argument is exactly `name`; the next one is the value.
            return args.next();
        } else if suffix.starts_with('=') {
            // This argument is `name=value`; get the value.
            // Strip leading `=`.
            return Some(suffix[1..].to_owned());
        }
    }
}

/// Returns the path to the `miri` binary
fn find_miri() -> PathBuf {
    if let Some(path) = env::var_os("MIRI") {
        return path.into();
    }
    let mut path = std::env::current_exe().expect("current executable path invalid");
    path.set_file_name("miri");
    path
}

fn miri() -> Command {
    Command::new(find_miri())
}

fn version_info() -> VersionMeta {
    VersionMeta::for_command(miri()).expect("failed to determine underlying rustc version of Miri")
}

fn cargo() -> Command {
    Command::new(env::var_os("CARGO").unwrap_or_else(|| OsString::from("cargo")))
}

fn xargo_check() -> Command {
    Command::new(env::var_os("XARGO_CHECK").unwrap_or_else(|| OsString::from("xargo-check")))
}

fn list_targets() -> impl Iterator<Item = cargo_metadata::Target> {
    // We need to get the manifest, and then the metadata, to enumerate targets.
    let manifest_path =
        get_arg_flag_value("--manifest-path").map(|m| Path::new(&m).canonicalize().unwrap());

    let mut cmd = cargo_metadata::MetadataCommand::new();
    if let Some(manifest_path) = &manifest_path {
        cmd.manifest_path(manifest_path);
    }
    let mut metadata = if let Ok(metadata) = cmd.exec() {
        metadata
    } else {
        show_error(format!("Could not obtain Cargo metadata; likely an ill-formed manifest"));
    };

    let current_dir = std::env::current_dir();

    let package_index = metadata
        .packages
        .iter()
        .position(|package| {
            let package_manifest_path = Path::new(&package.manifest_path);
            if let Some(manifest_path) = &manifest_path {
                package_manifest_path == manifest_path
            } else {
                let current_dir = current_dir.as_ref().expect("could not read current directory");
                let package_manifest_directory = package_manifest_path
                    .parent()
                    .expect("could not find parent directory of package manifest");
                package_manifest_directory == current_dir
            }
        })
        .unwrap_or_else(|| {
            show_error(format!(
                "this seems to be a workspace, which is not supported by `cargo miri`.\n\
                 Try to `cd` into the crate you want to test, and re-run `cargo miri` there."
            ))
        });
    let package = metadata.packages.remove(package_index);

    // Finally we got the list of targets to build
    package.targets.into_iter()
}

fn xargo_version() -> Option<(u32, u32, u32)> {
    let out = xargo_check().arg("--version").output().ok()?;
    if !out.status.success() {
        return None;
    }
    // Parse output. The first line looks like "xargo 0.3.12 (b004f1c 2018-12-13)".
    let line = out
        .stderr
        .lines()
        .nth(0)
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
    if !version_pieces.next().is_none() {
        panic!("malformed `xargo --version` output: more than three pieces in version");
    }
    Some((major, minor, patch))
}

fn ask_to_run(mut cmd: Command, ask: bool, text: &str) {
    // Disable interactive prompts in CI (GitHub Actions, Travis, AppVeyor, etc).
    if ask && env::var_os("CI").is_none() {
        let mut buf = String::new();
        print!("I will run `{:?}` to {}. Proceed? [Y/n] ", cmd, text);
        io::stdout().flush().unwrap();
        io::stdin().read_line(&mut buf).unwrap();
        match buf.trim().to_lowercase().as_ref() {
            // Proceed.
            "" | "y" | "yes" => {}
            "n" | "no" => show_error(format!("Aborting as per your request")),
            a => show_error(format!("I do not understand `{}`", a)),
        };
    } else {
        println!("Running `{:?}` to {}.", cmd, text);
    }

    if cmd.status().expect(&format!("failed to execute {:?}", cmd)).success().not() {
        show_error(format!("Failed to {}", text));
    }
}

/// Performs the setup required to make `cargo miri` work: Getting a custom-built libstd. Then sets
/// `MIRI_SYSROOT`. Skipped if `MIRI_SYSROOT` is already set, in which case we expect the user has
/// done all this already.
fn setup(subcommand: MiriCommand) {
    if std::env::var_os("MIRI_SYSROOT").is_some() {
        if subcommand == MiriCommand::Setup {
            println!("WARNING: MIRI_SYSROOT already set, not doing anything.")
        }
        return;
    }

    // Subcommands other than `setup` will do a setup if necessary, but
    // interactively confirm first.
    let ask_user = subcommand != MiriCommand::Setup;

    // First, we need xargo.
    if xargo_version().map_or(true, |v| v < XARGO_MIN_VERSION) {
        if std::env::var_os("XARGO_CHECK").is_some() {
            // The user manually gave us a xargo binary; don't do anything automatically.
            show_error(format!("Your xargo is too old; please upgrade to the latest version"))
        }
        let mut cmd = cargo();
        cmd.args(&["install", "xargo", "-f"]);
        ask_to_run(cmd, ask_user, "install a recent enough xargo");
    }

    // Determine where the rust sources are located.  `XARGO_RUST_SRC` env var trumps everything.
    let rust_src = match std::env::var_os("XARGO_RUST_SRC") {
        Some(path) => {
            let path = PathBuf::from(path);
            // Make path absolute if possible.
            path.canonicalize().unwrap_or(path)
        }
        None => {
            // Check for `rust-src` rustup component.
            let sysroot = miri()
                .args(&["--print", "sysroot"])
                .output()
                .expect("failed to determine sysroot")
                .stdout;
            let sysroot = std::str::from_utf8(&sysroot).unwrap();
            let sysroot = Path::new(sysroot.trim_end_matches('\n'));
            // Check for `$SYSROOT/lib/rustlib/src/rust/library`; test if that contains `std/Cargo.toml`.
            let rustup_src =
                sysroot.join("lib").join("rustlib").join("src").join("rust").join("library");
            if !rustup_src.join("std").join("Cargo.toml").exists() {
                // Ask the user to install the `rust-src` component, and use that.
                let mut cmd = Command::new("rustup");
                cmd.args(&["component", "add", "rust-src"]);
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
        show_error(format!("Given Rust source directory `{}` does not exist.", rust_src.display()));
    }

    // Next, we need our own libstd. Prepare a xargo project for that purpose.
    // We will do this work in whatever is a good cache dir for this platform.
    let dirs = directories::ProjectDirs::from("org", "rust-lang", "miri").unwrap();
    let dir = dirs.cache_dir();
    if !dir.exists() {
        fs::create_dir_all(&dir).unwrap();
    }
    // The interesting bit: Xargo.toml
    File::create(dir.join("Xargo.toml"))
        .unwrap()
        .write_all(
            br#"
[dependencies.std]
default_features = false
# We need the `panic_unwind` feature because we use the `unwind` panic strategy.
# Using `abort` works for libstd, but then libtest will not compile.
features = ["panic_unwind"]

[dependencies.test]
"#,
        )
        .unwrap();
    // The boring bits: a dummy project for xargo.
    // FIXME: With xargo-check, can we avoid doing this?
    File::create(dir.join("Cargo.toml"))
        .unwrap()
        .write_all(
            br#"
[package]
name = "miri-xargo"
description = "A dummy project for building libstd with xargo."
version = "0.0.0"

[lib]
path = "lib.rs"
"#,
        )
        .unwrap();
    File::create(dir.join("lib.rs")).unwrap();

    // Determine architectures.
    // We always need to set a target so rustc bootstrap can tell apart host from target crates.
    let host = version_info().host;
    let target = get_arg_flag_value("--target");
    let target = target.as_ref().unwrap_or(&host);
    // Now invoke xargo.
    let mut command = xargo_check();
    command.arg("check").arg("-q");
    command.arg("--target").arg(target);
    command.current_dir(&dir);
    command.env("XARGO_HOME", &dir);
    command.env("XARGO_RUST_SRC", &rust_src);
    // Use Miri as rustc to build a libstd compatible with us (and use the right flags).
    // However, when we are running in bootstrap, we cannot just overwrite `RUSTC`,
    // because we still need bootstrap to distinguish between host and target crates.
    // In that case we overwrite `RUSTC_REAL` instead which determines the rustc used
    // for target crates.
    if env::var_os("RUSTC_STAGE").is_some() {
        command.env("RUSTC_REAL", find_miri());
    } else {
        command.env("RUSTC", find_miri());
    }
    command.env("MIRI_BE_RUSTC", "1");
    // Make sure there are no other wrappers or flags getting in our way
    // (Cc https://github.com/rust-lang/miri/issues/1421).
    // This is consistent with normal `cargo build` that does not apply `RUSTFLAGS`
    // to the sysroot either.
    command.env_remove("RUSTC_WRAPPER");
    command.env_remove("RUSTFLAGS");
    // Finally run it!
    if command.status().expect("failed to run xargo").success().not() {
        show_error(format!("Failed to run xargo"));
    }

    // That should be it! But we need to figure out where xargo built stuff.
    // Unfortunately, it puts things into a different directory when the
    // architecture matches the host.
    let sysroot = if target == &host { dir.join("HOST") } else { PathBuf::from(dir) };
    std::env::set_var("MIRI_SYSROOT", &sysroot); // pass the env var to the processes we spawn, which will turn it into "--sysroot" flags
    // Figure out what to print.
    let print_sysroot = subcommand == MiriCommand::Setup && has_arg_flag("--print-sysroot"); // whether we just print the sysroot path
    if print_sysroot {
        // Print just the sysroot and nothing else; this way we do not need any escaping.
        println!("{}", sysroot.display());
    } else if subcommand == MiriCommand::Setup {
        println!("A libstd for Miri is now available in `{}`.", sysroot.display());
    }
}

fn in_cargo_miri() {
    let (subcommand, skip) = match std::env::args().nth(2).as_deref() {
        Some("test") => (MiriCommand::Test, 3),
        Some("run") => (MiriCommand::Run, 3),
        Some("setup") => (MiriCommand::Setup, 3),
        // Default command, if there is an option or nothing.
        Some(s) if s.starts_with("-") => (MiriCommand::Run, 2),
        None => (MiriCommand::Run, 2),
        // Invalid command.
        Some(s) => show_error(format!("Unknown command `{}`", s)),
    };
    let verbose = has_arg_flag("-v");

    // We always setup.
    setup(subcommand);
    if subcommand == MiriCommand::Setup {
        // Stop here.
        return;
    }

    // Now run the command.
    for target in list_targets() {
        let mut args = std::env::args().skip(skip);
        let kind = target
            .kind
            .get(0)
            .expect("badly formatted cargo metadata: target::kind is an empty array");
        // Now we run `cargo check $FLAGS $ARGS`, giving the user the
        // change to add additional arguments. `FLAGS` is set to identify
        // this target.  The user gets to control what gets actually passed to Miri.
        let mut cmd = cargo();
        cmd.arg("check");
        match (subcommand, kind.as_str()) {
            (MiriCommand::Run, "bin") => {
                // FIXME: we just run all the binaries here.
                // We should instead support `cargo miri --bin foo`.
                cmd.arg("--bin").arg(target.name);
            }
            (MiriCommand::Test, "test") => {
                cmd.arg("--test").arg(target.name);
            }
            (MiriCommand::Test, "lib") => {
                // There can be only one lib.
                cmd.arg("--lib").arg("--profile").arg("test");
            }
            (MiriCommand::Test, "bin") => {
                cmd.arg("--bin").arg(target.name).arg("--profile").arg("test");
            }
            // The remaining targets we do not even want to build.
            _ => continue,
        }
        // Forward user-defined `cargo` args until first `--`.
        while let Some(arg) = args.next() {
            if arg == "--" {
                break;
            }
            cmd.arg(arg);
        }
        // We want to always run `cargo` with `--target`. This later helps us detect
        // which crates are proc-macro/build-script (host crates) and which crates are
        // needed for the program itself.
        if get_arg_flag_value("--target").is_none() {
            // When no `--target` is given, default to the host.
            cmd.arg("--target");
            cmd.arg(version_info().host);
        }

        // Serialize the remaining args into a special environemt variable.
        // This will be read by `inside_cargo_rustc` when we go to invoke
        // our actual target crate (the binary or the test we are running).
        // Since we're using "cargo check", we have no other way of passing
        // these arguments.
        let args_vec: Vec<String> = args.collect();
        cmd.env("MIRI_ARGS", serde_json::to_string(&args_vec).expect("failed to serialize args"));

        // Set `RUSTC_WRAPPER` to ourselves.  Cargo will prepend that binary to its usual invocation,
        // i.e., the first argument is `rustc` -- which is what we use in `main` to distinguish
        // the two codepaths. (That extra argument is why we prefer this over setting `RUSTC`.)
        if env::var_os("RUSTC_WRAPPER").is_some() {
            println!("WARNING: Ignoring existing `RUSTC_WRAPPER` environment variable, Miri does not support wrapping.");
        }
        let path = std::env::current_exe().expect("current executable path invalid");
        cmd.env("RUSTC_WRAPPER", path);
        if verbose {
            cmd.env("MIRI_VERBOSE", ""); // this makes `inside_cargo_rustc` verbose.
            eprintln!("+ {:?}", cmd);
        }

        let exit_status =
            cmd.spawn().expect("could not run cargo").wait().expect("failed to wait for cargo?");

        if !exit_status.success() {
            std::process::exit(exit_status.code().unwrap_or(-1))
        }
    }
}

fn inside_cargo_rustc() {
    /// Determines if we are being invoked (as rustc) to build a crate for
    /// the "target" architecture, in contrast to the "host" architecture.
    /// Host crates are for build scripts and proc macros and still need to
    /// be built like normal; target crates need to be built for or interpreted
    /// by Miri.
    ///
    /// Currently, we detect this by checking for "--target=", which is
    /// never set for host crates. This matches what rustc bootstrap does,
    /// which hopefully makes it "reliable enough". This relies on us always
    /// invoking cargo itself with `--target`, which `in_cargo_miri` ensures.
    fn is_target_crate() -> bool {
        get_arg_flag_value("--target").is_some()
    }

    /// Returns whether or not Cargo invoked the wrapper (this binary) to compile
    /// the final, binary crate (either a test for 'cargo test', or a binary for 'cargo run')
    /// Cargo does not give us this information directly, so we need to check
    /// various command-line flags.
    fn is_runnable_crate() -> bool {
        let is_bin = get_arg_flag_value("--crate-type").as_deref() == Some("bin");
        let is_test = has_arg_flag("--test");
        is_bin || is_test
    }

    let verbose = std::env::var_os("MIRI_VERBOSE").is_some();
    let target_crate = is_target_crate();

    let mut cmd = miri();
    // Forward arguments.
    cmd.args(std::env::args().skip(2)); // skip `cargo-miri rustc`

    // We make sure to only specify our custom Xargo sysroot for target crates - that is,
    // crates which are needed for interpretation by Miri. proc-macros and build scripts
    // should use the default sysroot.
    if target_crate {
        let sysroot =
            env::var_os("MIRI_SYSROOT").expect("The wrapper should have set MIRI_SYSROOT");
        cmd.arg("--sysroot");
        cmd.arg(sysroot);
    }

    // If this is a runnable target crate, we want Miri to start interpretation;
    // otherwise we want Miri to behave like rustc and build the crate as usual.
    if target_crate && is_runnable_crate() {
        // This is the binary or test crate that we want to interpret under Miri.
        // (Testing `target_crate` is needed to exclude build scripts.)
        // We deserialize the arguments that are meant for Miri from the special environment
        // variable "MIRI_ARGS", and feed them to the 'miri' binary.
        //
        // `env::var` is okay here, well-formed JSON is always UTF-8.
        let magic = std::env::var("MIRI_ARGS").expect("missing MIRI_ARGS");
        let miri_args: Vec<String> =
            serde_json::from_str(&magic).expect("failed to deserialize MIRI_ARGS");
        cmd.args(miri_args);
    } else {
        // We want to compile, not interpret.
        cmd.env("MIRI_BE_RUSTC", "1");
    };

    // Run it.
    if verbose {
        eprintln!("+ {:?}", cmd);
    }
    match cmd.status() {
        Ok(exit) =>
            if !exit.success() {
                std::process::exit(exit.code().unwrap_or(42));
            },
        Err(e) => panic!("error running {:?}:\n{:?}", cmd, e),
    }
}

fn main() {
    // Check for version and help flags even when invoked as `cargo-miri`.
    if has_arg_flag("--help") || has_arg_flag("-h") {
        show_help();
        return;
    }
    if has_arg_flag("--version") || has_arg_flag("-V") {
        show_version();
        return;
    }

    if let Some("miri") = std::env::args().nth(1).as_deref() {
        // This arm is for when `cargo miri` is called. We call `cargo check` for each applicable target,
        // but with the `RUSTC` env var set to the `cargo-miri` binary so that we come back in the other branch,
        // and dispatch the invocations to `rustc` and `miri`, respectively.
        in_cargo_miri();
    } else if let Some("rustc") = std::env::args().nth(1).as_deref() {
        // This arm is executed when `cargo-miri` runs `cargo check` with the `RUSTC_WRAPPER` env var set to itself:
        // dependencies get dispatched to `rustc`, the final test/binary to `miri`.
        inside_cargo_rustc();
    } else {
        show_error(format!(
            "`cargo-miri` must be called with either `miri` or `rustc` as first argument."
        ))
    }
}
