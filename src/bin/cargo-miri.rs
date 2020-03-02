#![feature(inner_deref)]

use std::fs::{self, File};
use std::io::{self, BufRead, Write};
use std::ops::Not;
use std::path::{Path, PathBuf};
use std::process::Command;

const XARGO_MIN_VERSION: (u32, u32, u32) = (0, 3, 19);

const CARGO_MIRI_HELP: &str = r#"Interprets bin crates and tests in Miri

Usage:
    cargo miri [subcommand] [options] [--] [<miri opts>...] [--] [<program opts>...]

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
    let mut path = std::env::current_exe().expect("current executable path invalid");
    path.set_file_name("miri");
    path
}

fn cargo() -> Command {
    if let Ok(val) = std::env::var("CARGO") {
        // Bootstrap tells us where to find cargo
        Command::new(val)
    } else {
        Command::new("cargo")
    }
}

fn xargo_check() -> Command {
    if let Ok(val) = std::env::var("XARGO_CHECK") {
        // Bootstrap tells us where to find xargo
        Command::new(val)
    } else {
        Command::new("xargo-check")
    }
}

fn list_targets() -> impl Iterator<Item = cargo_metadata::Target> {
    // We need to get the manifest, and then the metadata, to enumerate targets.
    let manifest_path =
        get_arg_flag_value("--manifest-path").map(|m| Path::new(&m).canonicalize().unwrap());

    let mut cmd = cargo_metadata::MetadataCommand::new();
    if let Some(ref manifest_path) = manifest_path {
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
            if let Some(ref manifest_path) = manifest_path {
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
                "This seems to be a workspace, which is not supported by cargo-miri"
            ))
        });
    let package = metadata.packages.remove(package_index);

    // Finally we got the list of targets to build
    package.targets.into_iter()
}

/// Make sure that the `miri` and `rustc` binary are from the same sysroot.
/// This can be violated e.g. when miri is locally built and installed with a different
/// toolchain than what is used when `cargo miri` is run.
fn test_sysroot_consistency() {
    fn get_sysroot(mut cmd: Command) -> PathBuf {
        let out = cmd
            .arg("--print")
            .arg("sysroot")
            .output()
            .expect("Failed to run rustc to get sysroot info");
        let stdout = String::from_utf8(out.stdout).expect("stdout is not valid UTF-8");
        let stderr = String::from_utf8(out.stderr).expect("stderr is not valid UTF-8");
        assert!(
            out.status.success(),
            "Bad status code {} when getting sysroot info via {:?}.\nstdout:\n{}\nstderr:\n{}",
            out.status,
            cmd,
            stdout,
            stderr,
        );
        let stdout = stdout.trim();
        PathBuf::from(stdout)
            .canonicalize()
            .unwrap_or_else(|_| panic!("Failed to canonicalize sysroot: {}", stdout))
    }

    // Do not check sysroots if we got built as part of a Rust distribution.
    // During `bootstrap`, the sysroot does not match anyway, and then some distros
    // play symlink tricks so the sysroots may be different even for the final stage
    // (see <https://github.com/mozilla/nixpkgs-mozilla/issues/198>).
    if option_env!("RUSTC_STAGE").is_some() {
        return;
    }

    let rustc_sysroot = get_sysroot(Command::new("rustc"));
    let miri_sysroot = get_sysroot(Command::new(find_miri()));

    if rustc_sysroot != miri_sysroot {
        show_error(format!(
            "miri was built for a different sysroot than the rustc in your current toolchain.\n\
             Make sure you use the same toolchain to run miri that you used to build it!\n\
             rustc sysroot: `{}`\n\
             miri sysroot: `{}`",
            rustc_sysroot.display(),
            miri_sysroot.display()
        ));
    }
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
    if ask {
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
fn setup(ask_user: bool) {
    if std::env::var("MIRI_SYSROOT").is_ok() {
        if !ask_user {
            println!("WARNING: MIRI_SYSROOT already set, not doing anything.")
        }
        return;
    }

    // First, we need xargo.
    if xargo_version().map_or(true, |v| v < XARGO_MIN_VERSION) {
        if std::env::var("XARGO_CHECK").is_ok() {
            // The user manually gave us a xargo binary; don't do anything automatically.
            show_error(format!("Your xargo is too old; please upgrade to the latest version"))
        }
        let mut cmd = cargo();
        cmd.args(&["install", "xargo", "-f"]);
        ask_to_run(cmd, ask_user, "install a recent enough xargo");
    }

    // Determine where the rust sources are located.  `XARGO_RUST_SRC` env var trumps everything.
    let rust_src = match std::env::var("XARGO_RUST_SRC") {
        Ok(val) => PathBuf::from(val),
        Err(_) => {
            // Check for `rust-src` rustup component.
            let sysroot = Command::new("rustc")
                .args(&["--print", "sysroot"])
                .output()
                .expect("failed to get rustc sysroot")
                .stdout;
            let sysroot = std::str::from_utf8(&sysroot).unwrap();
            let sysroot = Path::new(sysroot.trim_end_matches('\n'));
            // First try: `$SYSROOT/lib/rustlib/src/rust`; test if that contains `Cargo.lock`.
            let rustup_src = sysroot.join("lib").join("rustlib").join("src").join("rust");
            let base_dir = if rustup_src.join("Cargo.lock").exists() {
                // Just use this.
                rustup_src
            } else {
                // Maybe this is a local toolchain built with `x.py` and linked into `rustup`?
                // Second try: `$SYSROOT/../../..`; test if that contains `x.py`.
                let local_src = sysroot.parent().and_then(Path::parent).and_then(Path::parent);
                match local_src {
                    Some(local_src) if local_src.join("x.py").exists() => {
                        // Use this.
                        PathBuf::from(local_src)
                    }
                    _ => {
                        // Fallback: Ask the user to install the `rust-src` component, and use that.
                        let mut cmd = Command::new("rustup");
                        cmd.args(&["component", "add", "rust-src"]);
                        ask_to_run(
                            cmd,
                            ask_user,
                            "install the rustc-src component for the selected toolchain",
                        );
                        rustup_src
                    }
                }
            };
            base_dir.join("src") // Xargo wants the src-subdir
        }
    };
    if !rust_src.exists() {
        show_error(format!("Given Rust source directory `{}` does not exist.", rust_src.display()));
    }

    // Next, we need our own libstd. We will do this work in whatever is a good cache dir for this platform.
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
    // Prepare xargo invocation.
    let target = get_arg_flag_value("--target");
    let print_sysroot = !ask_user && has_arg_flag("--print-sysroot"); // whether we just print the sysroot path
    let mut command = xargo_check();
    command.arg("build").arg("-q");
    command.current_dir(&dir);
    command.env("RUSTFLAGS", miri::miri_default_args().join(" "));
    command.env("XARGO_HOME", &dir);
    command.env("XARGO_RUST_SRC", &rust_src);
    // Handle target flag.
    if let Some(ref target) = target {
        command.arg("--target").arg(&target);
    }
    // Finally run it!
    if command.status().expect("failed to run xargo").success().not() {
        show_error(format!("Failed to run xargo"));
    }

    // That should be it! But we need to figure out where xargo built stuff.
    // Unfortunately, it puts things into a different directory when the
    // architecture matches the host.
    let is_host = match target {
        None => true,
        Some(target) => target == rustc_version::version_meta().unwrap().host,
    };
    let sysroot = if is_host { dir.join("HOST") } else { PathBuf::from(dir) };
    std::env::set_var("MIRI_SYSROOT", &sysroot); // pass the env var to the processes we spawn, which will turn it into "--sysroot" flags
    if print_sysroot {
        // Print just the sysroot and nothing else; this way we do not need any escaping.
        println!("{}", sysroot.display());
    } else if !ask_user {
        println!("A libstd for Miri is now available in `{}`.", sysroot.display());
    }
}

fn main() {
    // Check for version and help flags even when invoked as `cargo-miri`.
    if std::env::args().any(|a| a == "--help" || a == "-h") {
        show_help();
        return;
    }
    if std::env::args().any(|a| a == "--version" || a == "-V") {
        show_version();
        return;
    }

    if let Some("miri") = std::env::args().nth(1).as_ref().map(AsRef::as_ref) {
        // This arm is for when `cargo miri` is called. We call `cargo check` for each applicable target,
        // but with the `RUSTC` env var set to the `cargo-miri` binary so that we come back in the other branch,
        // and dispatch the invocations to `rustc` and `miri`, respectively.
        in_cargo_miri();
    } else if let Some("rustc") = std::env::args().nth(1).as_ref().map(AsRef::as_ref) {
        // This arm is executed when `cargo-miri` runs `cargo check` with the `RUSTC_WRAPPER` env var set to itself:
        // dependencies get dispatched to `rustc`, the final test/binary to `miri`.
        inside_cargo_rustc();
    } else {
        show_error(format!("must be called with either `miri` or `rustc` as first argument."))
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

    // Some basic sanity checks
    test_sysroot_consistency();

    // We always setup.
    let ask = subcommand != MiriCommand::Setup;
    setup(ask);
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

        // Serialize the remaining args into a special environemt variable.
        // This will be read by `inside_cargo_rustc` when we go to invoke
        // our actual target crate (the binary or the test we are running).
        // Since we're using "cargo check", we have no other way of passing
        // these arguments.
        let args_vec: Vec<String> = args.collect();
        cmd.env("MIRI_ARGS", serde_json::to_string(&args_vec).expect("failed to serialize args"));

        // Set `RUSTC_WRAPPER` to ourselves.  Cargo will prepend that binary to its usual invocation,
        // i.e., the first argument is `rustc` -- which is what we use in `main` to distinguish
        // the two codepaths.
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
    /// Determines if we are being invoked (as rustc) to build a runnable
    /// executable. We run "cargo check", so this should only happen when
    /// we are trying to compile a build script or build script dependency,
    /// which actually needs to be executed on the host platform.
    ///
    /// Currently, we detect this by checking for "--emit=link",
    /// which indicates that Cargo instruced rustc to output
    /// a native object.
    fn is_target_crate() -> bool {
        // `--emit` is sometimes missing, e.g. cargo calls rustc for "--print".
        // That is definitely not a target crate.
        // If `--emit` is present, then host crates are built ("--emit=link,...),
        // while the rest is only checked.
        get_arg_flag_value("--emit").map_or(false, |emit| !emit.contains("link"))
    }

    /// Returns whether or not Cargo invoked the wrapper (this binary) to compile
    /// the final, target crate (either a test for 'cargo test', or a binary for 'cargo run')
    /// Cargo does not give us this information directly, so we need to check
    /// various command-line flags.
    fn is_runnable_crate() -> bool {
        let is_bin = get_arg_flag_value("--crate-type").as_deref() == Some("bin");
        let is_test = has_arg_flag("--test");

        // The final runnable (under Miri) crate will either be a binary crate
        // or a test crate. We make sure to exclude build scripts here, since
        // they are also build with "--crate-type bin"
        is_bin || is_test
    }

    let verbose = std::env::var("MIRI_VERBOSE").is_ok();
    let target_crate = is_target_crate();

    // Figure out which arguments we need to pass.
    let mut args: Vec<String> = std::env::args().skip(2).collect(); // skip `cargo-miri rustc`
    // We make sure to only specify our custom Xargo sysroot and
    // other args for target crates - that is, crates which are ultimately
    // going to get interpreted by Miri.
    if target_crate {
        let sysroot =
            std::env::var("MIRI_SYSROOT").expect("The wrapper should have set MIRI_SYSROOT");
        args.push("--sysroot".to_owned());
        args.push(sysroot);
        args.splice(0..0, miri::miri_default_args().iter().map(ToString::to_string));
    }

    // Figure out the binary we need to call. If this is a runnable target crate, we want to call
    // Miri to start interpretation; otherwise we want to call rustc to build the crate as usual.
    let mut command = if target_crate && is_runnable_crate() {
        // This is the 'target crate' - the binary or test crate that
        // we want to interpret under Miri. We deserialize the user-provided arguments
        // from the special environment variable "MIRI_ARGS", and feed them
        // to the 'miri' binary.
        let magic = std::env::var("MIRI_ARGS").expect("missing MIRI_ARGS");
        let mut user_args: Vec<String> =
            serde_json::from_str(&magic).expect("failed to deserialize MIRI_ARGS");
        args.append(&mut user_args);
        // Run this in Miri.
        Command::new(find_miri())
    } else {
        Command::new("rustc")
    };

    // Run it.
    command.args(&args);
    if verbose {
        eprintln!("+ {:?}", command);
    }

    match command.status() {
        Ok(exit) =>
            if !exit.success() {
                std::process::exit(exit.code().unwrap_or(42));
            },
        Err(ref e) => panic!("error running {:?}:\n{:?}", command, e),
    }
}
