#![feature(let_else)]
#![allow(clippy::useless_format, clippy::derive_partial_eq_without_eq)]

mod version;

use std::collections::HashMap;
use std::env;
use std::ffi::{OsStr, OsString};
use std::fmt::Write as _;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, BufWriter, Read, Write};
use std::iter::{self, TakeWhile};
use std::ops::Not;
use std::path::{Path, PathBuf};
use std::process::{self, Command};

use rustc_version::VersionMeta;
use serde::{Deserialize, Serialize};

use version::*;

const CARGO_MIRI_HELP: &str = r#"Runs binary crates and tests in Miri

Usage:
    cargo miri [subcommand] [<cargo options>...] [--] [<program/test suite options>...]

Subcommands:
    run, r                   Run binaries
    test, t                  Run tests
    nextest                  Run tests with nextest (requires cargo-nextest installed)
    setup                    Only perform automatic setup, but without asking questions (for getting a proper libstd)

The cargo options are exactly the same as for `cargo run` and `cargo test`, respectively.

Examples:
    cargo miri run
    cargo miri test -- test-suite-filter

    cargo miri setup --print sysroot
        This will print the path to the generated sysroot (and nothing else) on stdout.
        stderr will still contain progress information about how the build is doing.

"#;

#[derive(Clone, Debug)]
enum MiriCommand {
    /// Our own special 'setup' command.
    Setup,
    /// A command to be forwarded to cargo.
    Forward(String),
}

/// The information to run a crate with the given environment.
#[derive(Serialize, Deserialize)]
struct CrateRunEnv {
    /// The command-line arguments.
    args: Vec<String>,
    /// The environment.
    env: Vec<(OsString, OsString)>,
    /// The current working directory.
    current_dir: OsString,
    /// The contents passed via standard input.
    stdin: Vec<u8>,
}

impl CrateRunEnv {
    /// Gather all the information we need.
    fn collect(args: impl Iterator<Item = String>, capture_stdin: bool) -> Self {
        let args = args.collect();
        let env = env::vars_os().collect();
        let current_dir = env::current_dir().unwrap().into_os_string();

        let mut stdin = Vec::new();
        if capture_stdin {
            std::io::stdin().lock().read_to_end(&mut stdin).expect("cannot read stdin");
        }

        CrateRunEnv { args, env, current_dir, stdin }
    }
}

/// The information Miri needs to run a crate. Stored as JSON when the crate is "compiled".
#[derive(Serialize, Deserialize)]
enum CrateRunInfo {
    /// Run it with the given environment.
    RunWith(CrateRunEnv),
    /// Skip it as Miri does not support interpreting such kind of crates.
    SkipProcMacroTest,
}

impl CrateRunInfo {
    fn store(&self, filename: &Path) {
        let file = File::create(filename)
            .unwrap_or_else(|_| show_error(format!("cannot create `{}`", filename.display())));
        let file = BufWriter::new(file);
        serde_json::ser::to_writer(file, self)
            .unwrap_or_else(|_| show_error(format!("cannot write to `{}`", filename.display())));
    }
}

fn show_help() {
    println!("{}", CARGO_MIRI_HELP);
}

fn show_version() {
    let mut version = format!("miri {}", env!("CARGO_PKG_VERSION"));
    // Only use `option_env` on vergen variables to ensure the build succeeds
    // when vergen failed to find the git info.
    if let Some(sha) = option_env!("VERGEN_GIT_SHA_SHORT") {
        // This `unwrap` can never fail because if VERGEN_GIT_SHA_SHORT exists, then so does
        // VERGEN_GIT_COMMIT_DATE.
        #[allow(clippy::option_env_unwrap)]
        write!(&mut version, " ({} {})", sha, option_env!("VERGEN_GIT_COMMIT_DATE").unwrap())
            .unwrap();
    }
    println!("{}", version);
}

fn show_error(msg: String) -> ! {
    eprintln!("fatal error: {}", msg);
    std::process::exit(1)
}

/// Determines whether a `--flag` is present.
fn has_arg_flag(name: &str) -> bool {
    num_arg_flag(name) > 0
}

/// Determines how many times a `--flag` is present.
fn num_arg_flag(name: &str) -> usize {
    std::env::args().take_while(|val| val != "--").filter(|val| val == name).count()
}

/// Yields all values of command line flag `name` as `Ok(arg)`, and all other arguments except
/// the flag as `Err(arg)`. (The flag `name` itself is not yielded at all, only its values are.)
struct ArgSplitFlagValue<'a, I> {
    args: TakeWhile<I, fn(&String) -> bool>,
    name: &'a str,
}

impl<'a, I: Iterator<Item = String>> ArgSplitFlagValue<'a, I> {
    fn new(args: I, name: &'a str) -> Self {
        Self {
            // Stop searching at `--`.
            args: args.take_while(|val| val != "--"),
            name,
        }
    }
}

impl<I: Iterator<Item = String>> Iterator for ArgSplitFlagValue<'_, I> {
    type Item = Result<String, String>;

    fn next(&mut self) -> Option<Self::Item> {
        let arg = self.args.next()?;
        if let Some(suffix) = arg.strip_prefix(self.name) {
            // Strip leading `name`.
            if suffix.is_empty() {
                // This argument is exactly `name`; the next one is the value.
                return self.args.next().map(Ok);
            } else if let Some(suffix) = suffix.strip_prefix('=') {
                // This argument is `name=value`; get the value.
                return Some(Ok(suffix.to_owned()));
            }
        }
        Some(Err(arg))
    }
}

/// Yields all values of command line flag `name`.
struct ArgFlagValueIter<'a>(ArgSplitFlagValue<'a, env::Args>);

impl<'a> ArgFlagValueIter<'a> {
    fn new(name: &'a str) -> Self {
        Self(ArgSplitFlagValue::new(env::args(), name))
    }
}

impl Iterator for ArgFlagValueIter<'_> {
    type Item = String;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Ok(value) = self.0.next()? {
                return Some(value);
            }
        }
    }
}

/// Gets the value of a `--flag`.
fn get_arg_flag_value(name: &str) -> Option<String> {
    ArgFlagValueIter::new(name).next()
}

fn forward_patched_extern_arg(args: &mut impl Iterator<Item = String>, cmd: &mut Command) {
    cmd.arg("--extern"); // always forward flag, but adjust filename:
    let path = args.next().expect("`--extern` should be followed by a filename");
    if let Some(lib) = path.strip_suffix(".rlib") {
        // If this is an rlib, make it an rmeta.
        cmd.arg(format!("{}.rmeta", lib));
    } else {
        // Some other extern file (e.g. a `.so`). Forward unchanged.
        cmd.arg(path);
    }
}

/// Escapes `s` in a way that is suitable for using it as a string literal in TOML syntax.
fn escape_for_toml(s: &str) -> String {
    // We want to surround this string in quotes `"`. So we first escape all quotes,
    // and also all backslashes (that are used to escape quotes).
    let s = s.replace('\\', r#"\\"#).replace('"', r#"\""#);
    format!("\"{}\"", s)
}

/// Returns the path to the `miri` binary
fn find_miri() -> PathBuf {
    if let Some(path) = env::var_os("MIRI") {
        return path.into();
    }
    let mut path = std::env::current_exe().expect("current executable path invalid");
    if cfg!(windows) {
        path.set_file_name("miri.exe");
    } else {
        path.set_file_name("miri");
    }
    path
}

fn miri() -> Command {
    Command::new(find_miri())
}

fn miri_for_host() -> Command {
    let mut cmd = miri();
    cmd.env("MIRI_BE_RUSTC", "host");
    cmd
}

fn version_info() -> VersionMeta {
    VersionMeta::for_command(miri_for_host())
        .expect("failed to determine underlying rustc version of Miri")
}

fn cargo() -> Command {
    Command::new(env::var_os("CARGO").unwrap_or_else(|| OsString::from("cargo")))
}

fn xargo_check() -> Command {
    Command::new(env::var_os("XARGO_CHECK").unwrap_or_else(|| OsString::from("xargo-check")))
}

/// Execute the command. If it fails, fail this process with the same exit code.
/// Otherwise, continue.
fn exec(mut cmd: Command) {
    let exit_status = cmd.status().expect("failed to run command");
    if exit_status.success().not() {
        std::process::exit(exit_status.code().unwrap_or(-1))
    }
}

/// Execute the command and pipe `input` into its stdin.
/// If it fails, fail this process with the same exit code.
/// Otherwise, continue.
fn exec_with_pipe(mut cmd: Command, input: &[u8]) {
    cmd.stdin(process::Stdio::piped());
    let mut child = cmd.spawn().expect("failed to spawn process");
    {
        let stdin = child.stdin.as_mut().expect("failed to open stdin");
        stdin.write_all(input).expect("failed to write out test source");
    }
    let exit_status = child.wait().expect("failed to run command");
    if exit_status.success().not() {
        std::process::exit(exit_status.code().unwrap_or(-1))
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

fn ask_to_run(mut cmd: Command, ask: bool, text: &str) {
    // Disable interactive prompts in CI (GitHub Actions, Travis, AppVeyor, etc).
    // Azure doesn't set `CI` though (nothing to see here, just Microsoft being Microsoft),
    // so we also check their `TF_BUILD`.
    let is_ci = env::var_os("CI").is_some() || env::var_os("TF_BUILD").is_some();
    if ask && !is_ci {
        let mut buf = String::new();
        print!("I will run `{:?}` to {}. Proceed? [Y/n] ", cmd, text);
        io::stdout().flush().unwrap();
        io::stdin().read_line(&mut buf).unwrap();
        match buf.trim().to_lowercase().as_ref() {
            // Proceed.
            "" | "y" | "yes" => {}
            "n" | "no" => show_error(format!("aborting as per your request")),
            a => show_error(format!("invalid answer `{}`", a)),
        };
    } else {
        eprintln!("Running `{:?}` to {}.", cmd, text);
    }

    if cmd.status().unwrap_or_else(|_| panic!("failed to execute {:?}", cmd)).success().not() {
        show_error(format!("failed to {}", text));
    }
}

/// Writes the given content to the given file *cross-process atomically*, in the sense that another
/// process concurrently reading that file will see either the old content or the new content, but
/// not some intermediate (e.g., empty) state.
///
/// We assume no other parts of this same process are trying to read or write that file.
fn write_to_file(filename: &Path, content: &str) {
    // Create a temporary file with the desired contents.
    let mut temp_filename = filename.as_os_str().to_os_string();
    temp_filename.push(&format!(".{}", std::process::id()));
    let mut temp_file = File::create(&temp_filename).unwrap();
    temp_file.write_all(content.as_bytes()).unwrap();
    drop(temp_file);

    // Move file to the desired location.
    fs::rename(temp_filename, filename).unwrap();
}

/// Performs the setup required to make `cargo miri` work: Getting a custom-built libstd. Then sets
/// `MIRI_SYSROOT`. Skipped if `MIRI_SYSROOT` is already set, in which case we expect the user has
/// done all this already.
fn setup(subcommand: &MiriCommand, host: &str, target: &str) {
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
            show_error(format!("xargo is too old; please upgrade to the latest version"))
        }
        let mut cmd = cargo();
        cmd.args(&["install", "xargo"]);
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
                .args(&["--print", "sysroot"])
                .output()
                .expect("failed to determine sysroot");
            if !output.status.success() {
                show_error(format!(
                    "Failed to determine sysroot; Miri said:\n{}",
                    String::from_utf8_lossy(&output.stderr).trim_end()
                ));
            }
            let sysroot = std::str::from_utf8(&output.stdout).unwrap();
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
        show_error(format!("given Rust source directory `{}` does not exist.", rust_src.display()));
    }
    if rust_src.file_name().and_then(OsStr::to_str) != Some("library") {
        show_error(format!(
            "given Rust source directory `{}` does not seem to be the `library` subdirectory of \
             a Rust source checkout.",
            rust_src.display()
        ));
    }

    // Next, we need our own libstd. Prepare a xargo project for that purpose.
    // We will do this work in whatever is a good cache dir for this platform.
    let dirs = directories::ProjectDirs::from("org", "rust-lang", "miri").unwrap();
    let dir = dirs.cache_dir();
    if !dir.exists() {
        fs::create_dir_all(&dir).unwrap();
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
    command.current_dir(&dir);
    command.env("XARGO_HOME", &dir);
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
        eprintln!("Preparing a sysroot for Miri...");
        if print_sysroot {
            // Be extra sure there is no noise on stdout.
            command.stdout(process::Stdio::null());
        }
    } else {
        // We want to be quiet, but still let the user know that something is happening.
        eprint!("Preparing a sysroot for Miri... ");
        command.stdout(process::Stdio::null());
        command.stderr(process::Stdio::null());
    }

    // Finally run it!
    if command.status().expect("failed to run xargo").success().not() {
        if only_setup {
            show_error(format!("failed to run xargo, see error details above"))
        } else {
            show_error(format!(
                "failed to run xargo; run `cargo miri setup` to see the error details"
            ))
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

#[derive(Deserialize)]
struct Metadata {
    target_directory: PathBuf,
    workspace_members: Vec<String>,
}

fn get_cargo_metadata() -> Metadata {
    let mut cmd = cargo();
    // `-Zunstable-options` is required by `--config`.
    cmd.args(["metadata", "--no-deps", "--format-version=1", "-Zunstable-options"]);
    // The `build.target-dir` config can be passed by `--config` flags, so forward them to
    // `cargo metadata`.
    let config_flag = "--config";
    for arg in ArgSplitFlagValue::new(
        env::args().skip(3), // skip the program name, "miri" and "run" / "test"
        config_flag,
    )
    // Only look at `Ok`
    .flatten()
    {
        cmd.arg(config_flag).arg(arg);
    }
    let mut child = cmd
        .stdin(process::Stdio::null())
        .stdout(process::Stdio::piped())
        .spawn()
        .expect("failed ro run `cargo metadata`");
    // Check this `Result` after `status.success()` is checked, so we don't print the error
    // to stderr if `cargo metadata` is also printing to stderr.
    let metadata: Result<Metadata, _> = serde_json::from_reader(child.stdout.take().unwrap());
    let status = child.wait().expect("failed to wait for `cargo metadata` to exit");
    if !status.success() {
        std::process::exit(status.code().unwrap_or(-1));
    }
    metadata.unwrap_or_else(|e| show_error(format!("invalid `cargo metadata` output: {}", e)))
}

/// Pulls all the crates in this workspace from the cargo metadata.
/// Workspace members are emitted like "miri 0.1.0 (path+file:///path/to/miri)"
/// Additionally, somewhere between cargo metadata and TyCtxt, '-' gets replaced with '_' so we
/// make that same transformation here.
fn local_crates(metadata: &Metadata) -> String {
    assert!(!metadata.workspace_members.is_empty());
    let mut local_crates = String::new();
    for member in &metadata.workspace_members {
        let name = member.split(' ').next().unwrap();
        let name = name.replace('-', "_");
        local_crates.push_str(&name);
        local_crates.push(',');
    }
    local_crates.pop(); // Remove the trailing ','

    local_crates
}

fn env_vars_from_cmd(cmd: &Command) -> Vec<(String, String)> {
    let mut envs = HashMap::new();
    for (key, value) in std::env::vars() {
        envs.insert(key, value);
    }
    for (key, value) in cmd.get_envs() {
        if let Some(value) = value {
            envs.insert(key.to_string_lossy().to_string(), value.to_string_lossy().to_string());
        } else {
            envs.remove(&key.to_string_lossy().to_string());
        }
    }
    let mut envs: Vec<_> = envs.into_iter().collect();
    envs.sort();
    envs
}

/// Debug-print a command that is going to be run.
fn debug_cmd(prefix: &str, verbose: usize, cmd: &Command) {
    if verbose == 0 {
        return;
    }
    // We only do a single `eprintln!` call to minimize concurrency interactions.
    let mut out = prefix.to_string();
    writeln!(out, " running command: env \\").unwrap();
    if verbose > 1 {
        // Print the full environment this will be called in.
        for (key, value) in env_vars_from_cmd(cmd) {
            writeln!(out, "{key}={value:?} \\").unwrap();
        }
    } else {
        // Print only what has been changed for this `cmd`.
        for (var, val) in cmd.get_envs() {
            if let Some(val) = val {
                writeln!(out, "{}={:?} \\", var.to_string_lossy(), val).unwrap();
            } else {
                writeln!(out, "--unset={}", var.to_string_lossy()).unwrap();
            }
        }
    }
    write!(out, "{cmd:?}").unwrap();
    eprintln!("{}", out);
}

fn phase_cargo_miri(mut args: impl Iterator<Item = String>) {
    // Check for version and help flags even when invoked as `cargo-miri`.
    if has_arg_flag("--help") || has_arg_flag("-h") {
        show_help();
        return;
    }
    if has_arg_flag("--version") || has_arg_flag("-V") {
        show_version();
        return;
    }

    // Require a subcommand before any flags.
    // We cannot know which of those flags take arguments and which do not,
    // so we cannot detect subcommands later.
    let Some(subcommand) = args.next() else {
        show_error(format!("`cargo miri` needs to be called with a subcommand (`run`, `test`)"));
    };
    let subcommand = match &*subcommand {
        "setup" => MiriCommand::Setup,
        "test" | "t" | "run" | "r" | "nextest" => MiriCommand::Forward(subcommand),
        _ =>
            show_error(format!(
                "`cargo miri` supports the following subcommands: `run`, `test`, `nextest`, and `setup`."
            )),
    };
    let verbose = num_arg_flag("-v");

    // Determine the involved architectures.
    let host = version_info().host;
    let target = get_arg_flag_value("--target");
    let target = target.as_ref().unwrap_or(&host);

    // We always setup.
    setup(&subcommand, &host, target);

    // Invoke actual cargo for the job, but with different flags.
    // We re-use `cargo test` and `cargo run`, which makes target and binary handling very easy but
    // requires some extra work to make the build check-only (see all the `--emit` hacks below).
    // <https://github.com/rust-lang/miri/pull/1540#issuecomment-693553191> describes an alternative
    // approach that uses `cargo check`, making that part easier but target and binary handling
    // harder.
    let cargo_miri_path = std::env::current_exe()
        .expect("current executable path invalid")
        .into_os_string()
        .into_string()
        .expect("current executable path is not valid UTF-8");
    let cargo_cmd = match subcommand {
        MiriCommand::Forward(s) => s,
        MiriCommand::Setup => return, // `cargo miri setup` stops here.
    };
    let metadata = get_cargo_metadata();
    let mut cmd = cargo();
    cmd.arg(cargo_cmd);

    // Forward all arguments before `--` other than `--target-dir` and its value to Cargo.
    let mut target_dir = None;
    for arg in ArgSplitFlagValue::new(&mut args, "--target-dir") {
        match arg {
            Ok(value) => {
                if target_dir.is_some() {
                    show_error(format!("`--target-dir` is provided more than once"));
                }
                target_dir = Some(value.into());
            }
            Err(arg) => {
                cmd.arg(arg);
            }
        }
    }
    // Detect the target directory if it's not specified via `--target-dir`.
    let target_dir = target_dir.get_or_insert_with(|| metadata.target_directory.clone());
    // Set `--target-dir` to `miri` inside the original target directory.
    target_dir.push("miri");
    cmd.arg("--target-dir").arg(target_dir);

    // Make sure the build target is explicitly set.
    // This is needed to make the `target.runner` settings do something,
    // and it later helps us detect which crates are proc-macro/build-script
    // (host crates) and which crates are needed for the program itself.
    if get_arg_flag_value("--target").is_none() {
        // No target given. Explicitly pick the host.
        cmd.arg("--target");
        cmd.arg(&host);
    }

    // Set ourselves as runner for al binaries invoked by cargo.
    // We use `all()` since `true` is not a thing in cfg-lang, but the empty conjunction is. :)
    let cargo_miri_path_for_toml = escape_for_toml(&cargo_miri_path);
    cmd.arg("--config")
        .arg(format!("target.'cfg(all())'.runner=[{cargo_miri_path_for_toml}, 'runner']"));

    // Forward all further arguments after `--` to cargo.
    cmd.arg("--").args(args);

    // Set `RUSTC_WRAPPER` to ourselves.  Cargo will prepend that binary to its usual invocation,
    // i.e., the first argument is `rustc` -- which is what we use in `main` to distinguish
    // the two codepaths. (That extra argument is why we prefer this over setting `RUSTC`.)
    if env::var_os("RUSTC_WRAPPER").is_some() {
        println!(
            "WARNING: Ignoring `RUSTC_WRAPPER` environment variable, Miri does not support wrapping."
        );
    }
    cmd.env("RUSTC_WRAPPER", &cargo_miri_path);
    // We are going to invoke `MIRI` for everything, not `RUSTC`.
    if env::var_os("RUSTC").is_some() && env::var_os("MIRI").is_none() {
        println!(
            "WARNING: Ignoring `RUSTC` environment variable; set `MIRI` if you want to control the binary used as the driver."
        );
    }
    // Build scripts (and also cargo: https://github.com/rust-lang/cargo/issues/10885) will invoke
    // `rustc` even when `RUSTC_WRAPPER` is set. To make sure everything is coherent, we want that
    // to be the Miri driver, but acting as rustc, on the target level. (Target, rather than host,
    // is needed for cross-interpretation situations.) This is not a perfect emulation of real rustc
    // (it might be unable to produce binaries since the sysroot is check-only), but it's as close
    // as we can get, and it's good enough for autocfg.
    //
    // In `main`, we need the value of `RUSTC` to distinguish RUSTC_WRAPPER invocations from rustdoc
    // or TARGET_RUNNER invocations, so we canonicalize it here to make it exceedingly unlikely that
    // there would be a collision with other invocations of cargo-miri (as rustdoc or as runner). We
    // explicitly do this even if RUSTC_STAGE is set, since for these builds we do *not* want the
    // bootstrap `rustc` thing in our way! Instead, we have MIRI_HOST_SYSROOT to use for host
    // builds.
    cmd.env("RUSTC", &fs::canonicalize(find_miri()).unwrap());
    cmd.env("MIRI_BE_RUSTC", "target"); // we better remember to *unset* this in the other phases!

    // Set rustdoc to us as well, so we can run doctests.
    cmd.env("RUSTDOC", &cargo_miri_path);

    cmd.env("MIRI_LOCAL_CRATES", local_crates(&metadata));
    if verbose > 0 {
        cmd.env("MIRI_VERBOSE", verbose.to_string()); // This makes the other phases verbose.
    }

    // Run cargo.
    debug_cmd("[cargo-miri miri]", verbose, &cmd);
    exec(cmd)
}

#[derive(Debug, Copy, Clone, PartialEq)]
enum RustcPhase {
    /// `rustc` called via `xargo` for sysroot build.
    Setup,
    /// `rustc` called by `cargo` for regular build.
    Build,
    /// `rustc` called by `rustdoc` for doctest.
    Rustdoc,
}

fn phase_rustc(mut args: impl Iterator<Item = String>, phase: RustcPhase) {
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
        let is_bin = get_arg_flag_value("--crate-type").as_deref().unwrap_or("bin") == "bin";
        let is_test = has_arg_flag("--test");
        is_bin || is_test
    }

    fn out_filename(prefix: &str, suffix: &str) -> PathBuf {
        if let Some(out_dir) = get_arg_flag_value("--out-dir") {
            let mut path = PathBuf::from(out_dir);
            path.push(format!(
                "{}{}{}{}",
                prefix,
                get_arg_flag_value("--crate-name").unwrap(),
                // This is technically a `-C` flag but the prefix seems unique enough...
                // (and cargo passes this before the filename so it should be unique)
                get_arg_flag_value("extra-filename").unwrap_or_default(),
                suffix,
            ));
            path
        } else {
            let out_file = get_arg_flag_value("-o").unwrap();
            PathBuf::from(out_file)
        }
    }

    // phase_cargo_miri set `MIRI_BE_RUSTC` for when build scripts directly invoke the driver;
    // however, if we get called back by cargo here, we'll carefully compute the right flags
    // ourselves, so we first un-do what the earlier phase did.
    env::remove_var("MIRI_BE_RUSTC");

    let verbose = std::env::var("MIRI_VERBOSE")
        .map_or(0, |verbose| verbose.parse().expect("verbosity flag must be an integer"));
    let target_crate = is_target_crate();
    // Determine whether this is cargo/xargo invoking rustc to get some infos.
    let info_query = get_arg_flag_value("--print").is_some() || has_arg_flag("-vV");

    let store_json = |info: CrateRunInfo| {
        // Create a stub .d file to stop Cargo from "rebuilding" the crate:
        // https://github.com/rust-lang/miri/issues/1724#issuecomment-787115693
        // As we store a JSON file instead of building the crate here, an empty file is fine.
        let dep_info_name = out_filename("", ".d");
        if verbose > 0 {
            eprintln!("[cargo-miri rustc] writing stub dep-info to `{}`", dep_info_name.display());
        }
        File::create(dep_info_name).expect("failed to create fake .d file");

        let filename = out_filename("", "");
        if verbose > 0 {
            eprintln!("[cargo-miri rustc] writing run info to `{}`", filename.display());
        }
        info.store(&filename);
        // For Windows, do the same thing again with `.exe` appended to the filename.
        // (Need to do this here as cargo moves that "binary" to a different place before running it.)
        info.store(&out_filename("", ".exe"));
    };

    let runnable_crate = !info_query && is_runnable_crate();

    if runnable_crate && target_crate {
        assert!(
            phase != RustcPhase::Setup,
            "there should be no interpretation during sysroot build"
        );
        let inside_rustdoc = phase == RustcPhase::Rustdoc;
        // This is the binary or test crate that we want to interpret under Miri.
        // But we cannot run it here, as cargo invoked us as a compiler -- our stdin and stdout are not
        // like we want them.
        // Instead of compiling, we write JSON into the output file with all the relevant command-line flags
        // and environment variables; this is used when cargo calls us again in the CARGO_TARGET_RUNNER phase.
        let env = CrateRunEnv::collect(args, inside_rustdoc);

        // Rustdoc expects us to exit with an error code if the test is marked as `compile_fail`,
        // just creating the JSON file is not enough: we need to detect syntax errors,
        // so we need to run Miri with `MIRI_BE_RUSTC` for a check-only build.
        if inside_rustdoc {
            let mut cmd = miri();

            // Ensure --emit argument for a check-only build is present.
            // We cannot use the usual helpers since we need to check specifically in `env.args`.
            if let Some(i) = env.args.iter().position(|arg| arg.starts_with("--emit=")) {
                // For `no_run` tests, rustdoc passes a `--emit` flag; make sure it has the right shape.
                assert_eq!(env.args[i], "--emit=metadata");
            } else {
                // For all other kinds of tests, we can just add our flag.
                cmd.arg("--emit=metadata");
            }

            cmd.args(&env.args);
            cmd.env("MIRI_BE_RUSTC", "target");

            if verbose > 0 {
                eprintln!(
                    "[cargo-miri rustc inside rustdoc] captured input:\n{}",
                    std::str::from_utf8(&env.stdin).unwrap()
                );
                eprintln!("[cargo-miri rustc inside rustdoc] going to run:\n{:?}", cmd);
            }

            exec_with_pipe(cmd, &env.stdin);
        }

        store_json(CrateRunInfo::RunWith(env));

        return;
    }

    if runnable_crate && ArgFlagValueIter::new("--extern").any(|krate| krate == "proc_macro") {
        // This is a "runnable" `proc-macro` crate (unit tests). We do not support
        // interpreting that under Miri now, so we write a JSON file to (display a
        // helpful message and) skip it in the runner phase.
        store_json(CrateRunInfo::SkipProcMacroTest);
        return;
    }

    let mut cmd = miri();
    let mut emit_link_hack = false;
    // Arguments are treated very differently depending on whether this crate is
    // for interpretation by Miri, or for use by a build script / proc macro.
    if !info_query && target_crate {
        // Forward arguments, but remove "link" from "--emit" to make this a check-only build.
        let emit_flag = "--emit";
        while let Some(arg) = args.next() {
            if let Some(val) = arg.strip_prefix(emit_flag) {
                // Patch this argument. First, extract its value.
                let val =
                    val.strip_prefix('=').expect("`cargo` should pass `--emit=X` as one argument");
                let mut val: Vec<_> = val.split(',').collect();
                // Now make sure "link" is not in there, but "metadata" is.
                if let Some(i) = val.iter().position(|&s| s == "link") {
                    emit_link_hack = true;
                    val.remove(i);
                    if !val.iter().any(|&s| s == "metadata") {
                        val.push("metadata");
                    }
                }
                cmd.arg(format!("{}={}", emit_flag, val.join(",")));
            } else if arg == "--extern" {
                // Patch `--extern` filenames, since Cargo sometimes passes stub `.rlib` files:
                // https://github.com/rust-lang/miri/issues/1705
                forward_patched_extern_arg(&mut args, &mut cmd);
            } else {
                cmd.arg(arg);
            }
        }

        // During setup, patch the panic runtime for `libpanic_abort` (mirroring what bootstrap usually does).
        if phase == RustcPhase::Setup
            && get_arg_flag_value("--crate-name").as_deref() == Some("panic_abort")
        {
            cmd.arg("-C").arg("panic=abort");
        }
    } else {
        // For host crates (but not when we are just printing some info),
        // we might still have to set the sysroot.
        if !info_query {
            // When we're running `cargo-miri` from `x.py` we need to pass the sysroot explicitly
            // due to bootstrap complications.
            if let Some(sysroot) = std::env::var_os("MIRI_HOST_SYSROOT") {
                cmd.arg("--sysroot").arg(sysroot);
            }
        }

        // For host crates or when we are printing, just forward everything.
        cmd.args(args);
    }

    // We want to compile, not interpret. We still use Miri to make sure the compiler version etc
    // are the exact same as what is used for interpretation.
    // MIRI_DEFAULT_ARGS should not be used to build host crates, hence setting "target" or "host"
    // as the value here to help Miri differentiate them.
    cmd.env("MIRI_BE_RUSTC", if target_crate { "target" } else { "host" });

    // Run it.
    if verbose > 0 {
        eprintln!(
            "[cargo-miri rustc] target_crate={target_crate} runnable_crate={runnable_crate} info_query={info_query}"
        );
    }
    debug_cmd("[cargo-miri rustc]", verbose, &cmd);
    exec(cmd);

    // Create a stub .rlib file if "link" was requested by cargo.
    // This is necessary to prevent cargo from doing rebuilds all the time.
    if emit_link_hack {
        // Some platforms prepend "lib", some do not... let's just create both files.
        File::create(out_filename("lib", ".rlib")).expect("failed to create fake .rlib file");
        File::create(out_filename("", ".rlib")).expect("failed to create fake .rlib file");
        // Just in case this is a cdylib or staticlib, also create those fake files.
        File::create(out_filename("lib", ".so")).expect("failed to create fake .so file");
        File::create(out_filename("lib", ".a")).expect("failed to create fake .a file");
        File::create(out_filename("lib", ".dylib")).expect("failed to create fake .dylib file");
        File::create(out_filename("", ".dll")).expect("failed to create fake .dll file");
        File::create(out_filename("", ".lib")).expect("failed to create fake .lib file");
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
enum RunnerPhase {
    /// `cargo` is running a binary
    Cargo,
    /// `rustdoc` is running a binary
    Rustdoc,
}

fn phase_runner(mut binary_args: impl Iterator<Item = String>, phase: RunnerPhase) {
    // phase_cargo_miri set `MIRI_BE_RUSTC` for when build scripts directly invoke the driver;
    // however, if we get called back by cargo here, we'll carefully compute the right flags
    // ourselves, so we first un-do what the earlier phase did.
    env::remove_var("MIRI_BE_RUSTC");

    let verbose = std::env::var("MIRI_VERBOSE")
        .map_or(0, |verbose| verbose.parse().expect("verbosity flag must be an integer"));

    let binary = binary_args.next().unwrap();
    let file = File::open(&binary)
        .unwrap_or_else(|_| show_error(format!(
            "file {:?} not found or `cargo-miri` invoked incorrectly; please only invoke this binary through `cargo miri`", binary
        )));
    let file = BufReader::new(file);

    let info = serde_json::from_reader(file).unwrap_or_else(|_| {
        show_error(format!(
            "file {:?} contains outdated or invalid JSON; try `cargo clean`",
            binary
        ))
    });
    let info = match info {
        CrateRunInfo::RunWith(info) => info,
        CrateRunInfo::SkipProcMacroTest => {
            eprintln!(
                "Running unit tests of `proc-macro` crates is not currently supported by Miri."
            );
            return;
        }
    };

    let mut cmd = miri();

    // Set missing env vars. We prefer build-time env vars over run-time ones; see
    // <https://github.com/rust-lang/miri/issues/1661> for the kind of issue that fixes.
    for (name, val) in info.env {
        if let Some(old_val) = env::var_os(&name) {
            if old_val == val {
                // This one did not actually change, no need to re-set it.
                // (This keeps the `debug_cmd` below more manageable.)
                continue;
            } else if verbose > 0 {
                eprintln!(
                    "[cargo-miri runner] Overwriting run-time env var {:?}={:?} with build-time value {:?}",
                    name, old_val, val
                );
            }
        }
        cmd.env(name, val);
    }

    // Forward rustc arguments.
    // We need to patch "--extern" filenames because we forced a check-only
    // build without cargo knowing about that: replace `.rlib` suffix by
    // `.rmeta`.
    // We also need to remove `--error-format` as cargo specifies that to be JSON,
    // but when we run here, cargo does not interpret the JSON any more. `--json`
    // then also nees to be dropped.
    let mut args = info.args.into_iter();
    let error_format_flag = "--error-format";
    let json_flag = "--json";
    while let Some(arg) = args.next() {
        if arg == "--extern" {
            forward_patched_extern_arg(&mut args, &mut cmd);
        } else if let Some(suffix) = arg.strip_prefix(error_format_flag) {
            assert!(suffix.starts_with('='));
            // Drop this argument.
        } else if let Some(suffix) = arg.strip_prefix(json_flag) {
            assert!(suffix.starts_with('='));
            // Drop this argument.
        } else {
            cmd.arg(arg);
        }
    }
    // Respect `MIRIFLAGS`.
    if let Ok(a) = env::var("MIRIFLAGS") {
        // This code is taken from `RUSTFLAGS` handling in cargo.
        let args = a.split(' ').map(str::trim).filter(|s| !s.is_empty()).map(str::to_string);
        cmd.args(args);
    }

    // Then pass binary arguments.
    cmd.arg("--");
    cmd.args(binary_args);

    // Make sure we use the build-time working directory for interpreting Miri/rustc arguments.
    // But then we need to switch to the run-time one, which we instruct Miri do do by setting `MIRI_CWD`.
    cmd.current_dir(info.current_dir);
    cmd.env("MIRI_CWD", env::current_dir().unwrap());

    // Run it.
    debug_cmd("[cargo-miri runner]", verbose, &cmd);
    match phase {
        RunnerPhase::Rustdoc => exec_with_pipe(cmd, &info.stdin),
        RunnerPhase::Cargo => exec(cmd),
    }
}

fn phase_rustdoc(mut args: impl Iterator<Item = String>) {
    let verbose = std::env::var("MIRI_VERBOSE")
        .map_or(0, |verbose| verbose.parse().expect("verbosity flag must be an integer"));

    // phase_cargo_miri sets the RUSTDOC env var to ourselves, so we can't use that here;
    // just default to a straight-forward invocation for now:
    let mut cmd = Command::new("rustdoc");

    let extern_flag = "--extern";
    let runtool_flag = "--runtool";
    while let Some(arg) = args.next() {
        if arg == extern_flag {
            // Patch --extern arguments to use *.rmeta files, since phase_cargo_rustc only creates stub *.rlib files.
            forward_patched_extern_arg(&mut args, &mut cmd);
        } else if arg == runtool_flag {
            // An existing --runtool flag indicates cargo is running in cross-target mode, which we don't support.
            // Note that this is only passed when cargo is run with the unstable -Zdoctest-xcompile flag;
            // otherwise, we won't be called as rustdoc at all.
            show_error(format!("cross-interpreting doctests is not currently supported by Miri."));
        } else {
            cmd.arg(arg);
        }
    }

    // Doctests of `proc-macro` crates (and their dependencies) are always built for the host,
    // so we are not able to run them in Miri.
    if ArgFlagValueIter::new("--crate-type").any(|crate_type| crate_type == "proc-macro") {
        eprintln!("Running doctests of `proc-macro` crates is not currently supported by Miri.");
        return;
    }

    // For each doctest, rustdoc starts two child processes: first the test is compiled,
    // then the produced executable is invoked. We want to reroute both of these to cargo-miri,
    // such that the first time we'll enter phase_cargo_rustc, and phase_cargo_runner second.
    //
    // rustdoc invokes the test-builder by forwarding most of its own arguments, which makes
    // it difficult to determine when phase_cargo_rustc should run instead of phase_cargo_rustdoc.
    // Furthermore, the test code is passed via stdin, rather than a temporary file, so we need
    // to let phase_cargo_rustc know to expect that. We'll use this environment variable as a flag:
    cmd.env("MIRI_CALLED_FROM_RUSTDOC", "1");

    // The `--test-builder` and `--runtool` arguments are unstable rustdoc features,
    // which are disabled by default. We first need to enable them explicitly:
    cmd.arg("-Z").arg("unstable-options");

    // rustdoc needs to know the right sysroot.
    cmd.arg("--sysroot").arg(env::var_os("MIRI_SYSROOT").unwrap());
    // make sure the 'miri' flag is set for rustdoc
    cmd.arg("--cfg").arg("miri");

    // Make rustdoc call us back.
    let cargo_miri_path = std::env::current_exe().expect("current executable path invalid");
    cmd.arg("--test-builder").arg(&cargo_miri_path); // invoked by forwarding most arguments
    cmd.arg("--runtool").arg(&cargo_miri_path); // invoked with just a single path argument

    debug_cmd("[cargo-miri rustdoc]", verbose, &cmd);
    exec(cmd)
}

fn main() {
    // Rustc does not support non-UTF-8 arguments so we make no attempt either.
    // (We do support non-UTF-8 environment variables though.)
    let mut args = std::env::args();
    // Skip binary name.
    args.next().unwrap();

    // Dispatch to `cargo-miri` phase. There are four phases:
    // - When we are called via `cargo miri`, we run as the frontend and invoke the underlying
    //   cargo. We set RUSTDOC, RUSTC_WRAPPER and CARGO_TARGET_RUNNER to ourselves.
    // - When we are executed due to RUSTDOC, we run rustdoc and set both `--test-builder` and
    //   `--runtool` to ourselves.
    // - When we are executed due to RUSTC_WRAPPER (or as the rustdoc test builder), we build crates
    //   or store the flags of binary crates for later interpretation.
    // - When we are executed due to CARGO_TARGET_RUNNER (or as the rustdoc runtool), we start
    //   interpretation based on the flags that were stored earlier.
    //
    // Additionally, we also set ourselves as RUSTC when calling xargo to build the sysroot, which
    // has to be treated slightly differently than when we build regular crates.

    // Dispatch running as part of sysroot compilation.
    if env::var_os("MIRI_CALLED_FROM_XARGO").is_some() {
        phase_rustc(args, RustcPhase::Setup);
        return;
    }

    // The way rustdoc invokes rustc is indistuingishable from the way cargo invokes rustdoc by the
    // arguments alone. `phase_cargo_rustdoc` sets this environment variable to let us disambiguate.
    if env::var_os("MIRI_CALLED_FROM_RUSTDOC").is_some() {
        // ...however, we then also see this variable when rustdoc invokes us as the testrunner!
        // The runner is invoked as `$runtool ($runtool-arg)* output_file`;
        // since we don't specify any runtool-args, and rustdoc supplies multiple arguments to
        // the test-builder unconditionally, we can just check the number of remaining arguments:
        if args.len() == 1 {
            phase_runner(args, RunnerPhase::Rustdoc);
        } else {
            phase_rustc(args, RustcPhase::Rustdoc);
        }

        return;
    }

    let Some(first) = args.next() else {
        show_error(format!(
            "`cargo-miri` called without first argument; please only invoke this binary through `cargo miri`"
        ))
    };
    match first.as_str() {
        "miri" => phase_cargo_miri(args),
        "runner" => phase_runner(args, RunnerPhase::Cargo),
        arg if arg == env::var("RUSTC").unwrap() => {
            // If the first arg is equal to the RUSTC env ariable (which should be set at this
            // point), then we need to behave as rustc. This is the somewhat counter-intuitive
            // behavior of having both RUSTC and RUSTC_WRAPPER set
            // (see https://github.com/rust-lang/cargo/issues/10886).
            phase_rustc(args, RustcPhase::Build)
        }
        _ => {
            // Everything else must be rustdoc. But we need to get `first` "back onto the iterator",
            // it is some part of the rustdoc invocation.
            phase_rustdoc(iter::once(first).chain(args));
        }
    }
}
