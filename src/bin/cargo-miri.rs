#![feature(inner_deref)]

use std::fs::{self, File};
use std::io::{self, Write, BufRead};
use std::path::{PathBuf, Path};
use std::process::Command;
use std::ops::Not;

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

Other [options] are the same as `cargo rustc`.  Everything after the first "--" is
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
    println!("miri {} ({} {})",
        env!("CARGO_PKG_VERSION"), env!("VERGEN_SHA_SHORT"), env!("VERGEN_COMMIT_DATE"));
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

fn list_targets() -> impl Iterator<Item=cargo_metadata::Target> {
    // We need to get the manifest, and then the metadata, to enumerate targets.
    let manifest_path = get_arg_flag_value("--manifest-path").map(|m|
        Path::new(&m).canonicalize().unwrap()
    );

    let mut cmd = cargo_metadata::MetadataCommand::new();
    if let Some(ref manifest_path) = manifest_path {
        cmd.manifest_path(manifest_path);
    }
    let mut metadata = if let Ok(metadata) = cmd.exec() {
        metadata
    } else {
        show_error(format!("Could not obtain Cargo metadata"));
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
                let current_dir = current_dir.as_ref().expect(
                    "could not read current directory",
                );
                let package_manifest_directory = package_manifest_path.parent().expect(
                    "could not find parent directory of package manifest",
                );
                package_manifest_directory == current_dir
            }
        })
        .expect("could not find matching package");
    let package = metadata.packages.remove(package_index);

    // Finally we got the list of targets to build
    package.targets.into_iter()
}

/// Returns the path to the `miri` binary
fn find_miri() -> PathBuf {
    let mut path = std::env::current_exe().expect("current executable path invalid");
    path.set_file_name("miri");
    path
}

/// Make sure that the `miri` and `rustc` binary are from the same sysroot.
/// This can be violated e.g. when miri is locally built and installed with a different
/// toolchain than what is used when `cargo miri` is run.
fn test_sysroot_consistency() {
    fn get_sysroot(mut cmd: Command) -> PathBuf {
        let out = cmd.arg("--print").arg("sysroot")
            .output().expect("Failed to run rustc to get sysroot info");
        let stdout = String::from_utf8(out.stdout).expect("stdout is not valid UTF-8");
        let stderr = String::from_utf8(out.stderr).expect("stderr is not valid UTF-8");
        let stdout = stdout.trim();
        assert!(out.status.success(), "Bad status code when getting sysroot info.\nstdout:\n{}\nstderr:\n{}", stdout, stderr);
        PathBuf::from(stdout).canonicalize()
            .unwrap_or_else(|_| panic!("Failed to canonicalize sysroot: {}", stdout))
    }

    // We let the user skip this check if they really want to.
    // (`bootstrap` needs this because Miri gets built by the stage1 compiler
    // but run with the stage2 sysroot.)
    if std::env::var("MIRI_SKIP_SYSROOT_CHECK").is_ok() {
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
             rustc_sysroot.display(), miri_sysroot.display()
        ));
    }
}

fn cargo() -> Command {
    if let Ok(val) = std::env::var("CARGO") {
        // Bootstrap tells us where to find cargo
        Command::new(val)
    } else {
        Command::new("cargo")
    }
}

fn xargo() -> Command {
    if let Ok(val) = std::env::var("XARGO") {
        // Bootstrap tells us where to find xargo
        Command::new(val)
    } else {
        Command::new("xargo")
    }
}

fn xargo_version() -> Option<(u32, u32, u32)> {
    let out = xargo().arg("--version").output().ok()?;
    if !out.status.success() {
        return None;
    }
    // Parse output. The first line looks like "xargo 0.3.12 (b004f1c 2018-12-13)".
    let line = out.stderr.lines().nth(0)
        .expect("malformed `xargo --version` output: not at least one line")
        .expect("malformed `xargo --version` output: error reading first line");
    let (name, version) = {
        let mut split = line.split(' ');
        (split.next().expect("malformed `xargo --version` output: empty"),
         split.next().expect("malformed `xargo --version` output: not at least two words"))
    };
    if name != "xargo" {
        // This is some fork of xargo
        return None;
    }
    let mut version_pieces = version.split('.');
    let major = version_pieces.next()
        .expect("malformed `xargo --version` output: not a major version piece")
        .parse()
        .expect("malformed `xargo --version` output: major version is not an integer");
    let minor = version_pieces.next()
        .expect("malformed `xargo --version` output: not a minor version piece")
        .parse()
        .expect("malformed `xargo --version` output: minor version is not an integer");
    let patch = version_pieces.next()
        .expect("malformed `xargo --version` output: not a patch version piece")
        .parse()
        .expect("malformed `xargo --version` output: patch version is not an integer");
    if !version_pieces.next().is_none() {
        panic!("malformed `xargo --version` output: more than three pieces in version");
    }
    Some((major, minor, patch))
}

fn ask(question: &str) {
    let mut buf = String::new();
    print!("{} [Y/n] ", question);
    io::stdout().flush().unwrap();
    io::stdin().read_line(&mut buf).unwrap();
    match buf.trim().to_lowercase().as_ref() {
        // Proceed.
        "" | "y" | "yes" => {},
        "n" | "no" => show_error(format!("Aborting as per your request")),
        a => show_error(format!("I do not understand `{}`", a))
    };
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
    if xargo_version().map_or(true, |v| v < (0, 3, 15)) {
        if ask_user {
            ask("It seems you do not have a recent enough xargo installed. I will run `cargo install xargo -f`. Proceed?");
        } else {
            println!("Installing xargo: `cargo install xargo -f`");
        }

        if cargo().args(&["install", "xargo", "-f"]).status()
            .expect("failed to install xargo")
            .success().not()
        {
            show_error(format!("Failed to install xargo"));
        }
    }

    // Then, unless `XARGO_RUST_SRC` is set, we also need rust-src.
    // Let's see if it is already installed.
    if std::env::var("XARGO_RUST_SRC").is_err() {
        let sysroot = Command::new("rustc").args(&["--print", "sysroot"]).output()
            .expect("failed to get rustc sysroot")
            .stdout;
        let sysroot = std::str::from_utf8(&sysroot).unwrap();
        let src = Path::new(sysroot.trim_end_matches('\n')).join("lib").join("rustlib").join("src");
        if !src.exists() {
            if ask_user {
                ask("It seems you do not have the rust-src component installed. I will run `rustup component add rust-src` for the selected toolchain. Proceed?");
            } else {
                println!("Installing rust-src component: `rustup component add rust-src`");
            }
            if !Command::new("rustup").args(&["component", "add", "rust-src"]).status()
                .expect("failed to install rust-src component")
                .success()
            {
                show_error(format!("Failed to install rust-src component"));
            }
        }
    }

    // Next, we need our own libstd. We will do this work in whatever is a good cache dir for this platform.
    let dirs = directories::ProjectDirs::from("org", "rust-lang", "miri").unwrap();
    let dir = dirs.cache_dir();
    if !dir.exists() {
        fs::create_dir_all(&dir).unwrap();
    }
    // The interesting bit: Xargo.toml
    File::create(dir.join("Xargo.toml")).unwrap()
        .write_all(br#"
[dependencies.std]
default_features = false
# We need the `panic_unwind` feature because we use the `unwind` panic strategy.
# Using `abort` works for libstd, but then libtest will not compile.
features = ["panic_unwind"]

[dependencies.test]
stage = 1
        "#).unwrap();
    // The boring bits: a dummy project for xargo.
    File::create(dir.join("Cargo.toml")).unwrap()
        .write_all(br#"
[package]
name = "miri-xargo"
description = "A dummy project for building libstd with xargo."
version = "0.0.0"

[lib]
path = "lib.rs"
        "#).unwrap();
    File::create(dir.join("lib.rs")).unwrap();
    // Prepare xargo invocation.
    let target = get_arg_flag_value("--target");
    let print_env = !ask_user && has_arg_flag("--env"); // whether we just print the necessary environment variable
    let mut command = xargo();
    command.arg("build").arg("-q");
    command.current_dir(&dir);
    command.env("RUSTFLAGS", miri::miri_default_args().join(" "));
    command.env("XARGO_HOME", dir.to_str().unwrap());
    // In bootstrap, make sure we don't get debug assertons into our libstd.
    command.env("RUSTC_DEBUG_ASSERTIONS", "false");
    // Handle target flag.
    if let Some(ref target) = target {
        command.arg("--target").arg(&target);
    }
    // Finally run it!
    if command.status()
        .expect("failed to run xargo")
        .success().not()
    {
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
    if print_env {
        println!("MIRI_SYSROOT={}", sysroot.display());
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
        // This arm is for when `cargo miri` is called. We call `cargo rustc` for each applicable target,
        // but with the `RUSTC` env var set to the `cargo-miri` binary so that we come back in the other branch,
        // and dispatch the invocations to `rustc` and `miri`, respectively.
        in_cargo_miri();
    } else if let Some("rustc") = std::env::args().nth(1).as_ref().map(AsRef::as_ref) {
        // This arm is executed when `cargo-miri` runs `cargo rustc` with the `RUSTC_WRAPPER` env var set to itself:
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
        Some(s) => {
            show_error(format!("Unknown command `{}`", s))
        }
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
        let kind = target.kind.get(0).expect(
            "badly formatted cargo metadata: target::kind is an empty array",
        );
        // Now we run `cargo rustc $FLAGS $ARGS`, giving the user the
        // change to add additional arguments. `FLAGS` is set to identify
        // this target.  The user gets to control what gets actually passed to Miri.
        let mut cmd = cargo();
        cmd.arg("rustc");
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
        // Add user-defined args until first `--`.
        while let Some(arg) = args.next() {
            if arg == "--" {
                break;
            }
            cmd.arg(arg);
        }
        // Add `--` (to end the `cargo` flags), and then the user flags. We add markers around the
        // user flags to be able to identify them later.  "cargo rustc" adds more stuff after this,
        // so we have to mark both the beginning and the end.
        cmd
            .arg("--")
            .arg("cargo-miri-marker-begin")
            .args(args)
            .arg("cargo-miri-marker-end");
        let path = std::env::current_exe().expect("current executable path invalid");
        cmd.env("RUSTC_WRAPPER", path);
        if verbose {
            eprintln!("+ {:?}", cmd);
        }

        let exit_status = cmd
            .spawn()
            .expect("could not run cargo")
            .wait()
            .expect("failed to wait for cargo?");

        if !exit_status.success() {
            std::process::exit(exit_status.code().unwrap_or(-1))
        }
    }
}

fn inside_cargo_rustc() {
    let sysroot = std::env::var("MIRI_SYSROOT").expect("The wrapper should have set MIRI_SYSROOT");

    let rustc_args = std::env::args().skip(2); // skip `cargo rustc`
    let mut args: Vec<String> = rustc_args
        .chain(Some("--sysroot".to_owned()))
        .chain(Some(sysroot))
        .collect();
    args.splice(0..0, miri::miri_default_args().iter().map(ToString::to_string));

    // See if we can find the `cargo-miri` markers. Those only get added to the binary we want to
    // run. They also serve to mark the user-defined arguments, which we have to move all the way
    // to the end (they get added somewhere in the middle).
    let needs_miri = if let Some(begin) = args.iter().position(|arg| arg == "cargo-miri-marker-begin") {
        let end = args
            .iter()
            .position(|arg| arg == "cargo-miri-marker-end")
            .expect("cannot find end marker");
        // These mark the user arguments. We remove the first and last as they are the markers.
        let mut user_args = args.drain(begin..=end);
        assert_eq!(user_args.next().unwrap(), "cargo-miri-marker-begin");
        assert_eq!(user_args.next_back().unwrap(), "cargo-miri-marker-end");
        // Collect the rest and add it back at the end.
        let mut user_args = user_args.collect::<Vec<String>>();
        args.append(&mut user_args);
        // Run this in Miri.
        true
    } else {
        false
    };

    let mut command = if needs_miri {
        Command::new(find_miri())
    } else {
        Command::new("rustc")
    };
    command.args(&args);
    if has_arg_flag("-v") {
        eprintln!("+ {:?}", command);
    }

    match command.status() {
        Ok(exit) => {
            if !exit.success() {
                std::process::exit(exit.code().unwrap_or(42));
            }
        }
        Err(ref e) if needs_miri => panic!("error during miri run: {:?}", e),
        Err(ref e) => panic!("error during rustc call: {:?}", e),
    }
}
