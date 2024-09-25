#![allow(clippy::needless_question_mark)]

mod args;
mod commands;
mod util;

use std::ops::Range;

use anyhow::{Context, Result, anyhow, bail};

#[derive(Clone, Debug)]
pub enum Command {
    /// Installs the miri driver and cargo-miri.
    /// Sets up the rpath such that the installed binary should work in any
    /// working directory. Note that the binaries are placed in the `miri` toolchain
    /// sysroot, to prevent conflicts with other toolchains.
    Install {
        /// Flags that are passed through to `cargo install`.
        flags: Vec<String>,
    },
    /// Just build miri.
    Build {
        /// Flags that are passed through to `cargo build`.
        flags: Vec<String>,
    },
    /// Just check miri.
    Check {
        /// Flags that are passed through to `cargo check`.
        flags: Vec<String>,
    },
    /// Build miri, set up a sysroot and then run the test suite.
    Test {
        bless: bool,
        /// The cross-interpretation target.
        /// If none then the host is the target.
        target: Option<String>,
        /// Flags that are passed through to the test harness.
        flags: Vec<String>,
    },
    /// Build miri, set up a sysroot and then run the driver with the given <flags>.
    /// (Also respects MIRIFLAGS environment variable.)
    Run {
        dep: bool,
        verbose: bool,
        many_seeds: Option<Range<u32>>,
        target: Option<String>,
        edition: Option<String>,
        /// Flags that are passed through to `miri`.
        flags: Vec<String>,
    },
    /// Build documentation
    Doc {
        /// Flags that are passed through to `cargo doc`.
        flags: Vec<String>,
    },
    /// Format all sources and tests.
    Fmt {
        /// Flags that are passed through to `rustfmt`.
        flags: Vec<String>,
    },
    /// Runs clippy on all sources.
    Clippy {
        /// Flags that are passed through to `cargo clippy`.
        flags: Vec<String>,
    },
    /// Runs the benchmarks from bench-cargo-miri in hyperfine. hyperfine needs to be installed.
    Bench {
        target: Option<String>,
        /// List of benchmarks to run. By default all benchmarks are run.
        benches: Vec<String>,
    },
    /// Update and activate the rustup toolchain 'miri' to the commit given in the
    /// `rust-version` file.
    /// `rustup-toolchain-install-master` must be installed for this to work. Any extra
    /// flags are passed to `rustup-toolchain-install-master`.
    Toolchain { flags: Vec<String> },
    /// Pull and merge Miri changes from the rustc repo. Defaults to fetching the latest
    /// rustc commit. The fetched commit is stored in the `rust-version` file, so the
    /// next `./miri toolchain` will install the rustc that just got pulled.
    RustcPull { commit: Option<String> },
    /// Push Miri changes back to the rustc repo. This will pull a copy of the rustc
    /// history into the Miri repo, unless you set the RUSTC_GIT env var to an existing
    /// clone of the rustc repo.
    RustcPush { github_user: String, branch: String },
}

const HELP: &str = r#"  COMMANDS

./miri build <flags>:
Just build miri. <flags> are passed to `cargo build`.

./miri check <flags>:
Just check miri. <flags> are passed to `cargo check`.

./miri test [--bless] [--target <target>] <flags>:
Build miri, set up a sysroot and then run the test suite.
<flags> are passed to the test harness.

./miri run [--dep] [-v|--verbose] [--many-seeds|--many-seeds=..to|--many-seeds=from..to] <flags>:
Build miri, set up a sysroot and then run the driver with the given <flags>.
(Also respects MIRIFLAGS environment variable.)
If `--many-seeds` is present, Miri is run many times in parallel with different seeds.
The range defaults to `0..64`.

./miri fmt <flags>:
Format all sources and tests. <flags> are passed to `rustfmt`.

./miri clippy <flags>:
Runs clippy on all sources. <flags> are passed to `cargo clippy`.

./miri cargo <flags>:
Runs just `cargo <flags>` with the Miri-specific environment variables.
Mainly meant to be invoked by rust-analyzer.

./miri install <flags>:
Installs the miri driver and cargo-miri. <flags> are passed to `cargo
install`. Sets up the rpath such that the installed binary should work in any
working directory. Note that the binaries are placed in the `miri` toolchain
sysroot, to prevent conflicts with other toolchains.

./miri bench [--target <target>] <benches>:
Runs the benchmarks from bench-cargo-miri in hyperfine. hyperfine needs to be installed.
<benches> can explicitly list the benchmarks to run; by default, all of them are run.

./miri toolchain <flags>:
Update and activate the rustup toolchain 'miri' to the commit given in the
`rust-version` file.
`rustup-toolchain-install-master` must be installed for this to work. Any extra
flags are passed to `rustup-toolchain-install-master`.

./miri rustc-pull <commit>:
Pull and merge Miri changes from the rustc repo. Defaults to fetching the latest
rustc commit. The fetched commit is stored in the `rust-version` file, so the
next `./miri toolchain` will install the rustc that just got pulled.

./miri rustc-push <github user> [<branch>]:
Push Miri changes back to the rustc repo. This will pull a copy of the rustc
history into the Miri repo, unless you set the RUSTC_GIT env var to an existing
clone of the rustc repo. The branch defaults to `miri-sync`.

  ENVIRONMENT VARIABLES

MIRI_SYSROOT:
If already set, the "sysroot setup" step is skipped.

CARGO_EXTRA_FLAGS:
Pass extra flags to all cargo invocations. (Ignored by `./miri cargo`.)"#;

fn main() -> Result<()> {
    // We are hand-rolling our own argument parser, since `clap` can't express what we need
    // (https://github.com/clap-rs/clap/issues/5055).
    let mut args = args::Args::new();
    let command = match args.next_raw().as_deref() {
        Some("build") => Command::Build { flags: args.remainder() },
        Some("check") => Command::Check { flags: args.remainder() },
        Some("doc") => Command::Doc { flags: args.remainder() },
        Some("test") => {
            let mut target = None;
            let mut bless = false;
            let mut flags = Vec::new();
            loop {
                if args.get_long_flag("bless")? {
                    bless = true;
                } else if let Some(val) = args.get_long_opt("target")? {
                    target = Some(val);
                } else if let Some(flag) = args.get_other() {
                    flags.push(flag);
                } else {
                    break;
                }
            }
            Command::Test { bless, flags, target }
        }
        Some("run") => {
            let mut dep = false;
            let mut verbose = false;
            let mut many_seeds = None;
            let mut target = None;
            let mut edition = None;
            let mut flags = Vec::new();
            loop {
                if args.get_long_flag("dep")? {
                    dep = true;
                } else if args.get_long_flag("verbose")? || args.get_short_flag('v')? {
                    verbose = true;
                } else if let Some(val) = args.get_long_opt_with_default("many-seeds", "0..64")? {
                    let (from, to) = val.split_once("..").ok_or_else(|| {
                        anyhow!("invalid format for `--many-seeds`: expected `from..to`")
                    })?;
                    let from: u32 = if from.is_empty() {
                        0
                    } else {
                        from.parse().context("invalid `from` in `--many-seeds=from..to")?
                    };
                    let to: u32 = to.parse().context("invalid `to` in `--many-seeds=from..to")?;
                    many_seeds = Some(from..to);
                } else if let Some(val) = args.get_long_opt("target")? {
                    target = Some(val);
                } else if let Some(val) = args.get_long_opt("edition")? {
                    edition = Some(val);
                } else if let Some(flag) = args.get_other() {
                    flags.push(flag);
                } else {
                    break;
                }
            }
            Command::Run { dep, verbose, many_seeds, target, edition, flags }
        }
        Some("fmt") => Command::Fmt { flags: args.remainder() },
        Some("clippy") => Command::Clippy { flags: args.remainder() },
        Some("install") => Command::Install { flags: args.remainder() },
        Some("bench") => {
            let mut target = None;
            let mut benches = Vec::new();
            loop {
                if let Some(val) = args.get_long_opt("target")? {
                    target = Some(val);
                } else if let Some(flag) = args.get_other() {
                    benches.push(flag);
                } else {
                    break;
                }
            }
            Command::Bench { target, benches }
        }
        Some("toolchain") => Command::Toolchain { flags: args.remainder() },
        Some("rustc-pull") => {
            let commit = args.next_raw();
            if args.next_raw().is_some() {
                bail!("Too many arguments for `./miri rustc-pull`");
            }
            Command::RustcPull { commit }
        }
        Some("rustc-push") => {
            let github_user = args.next_raw().ok_or_else(|| {
                anyhow!("Missing first argument for `./miri rustc-push GITHUB_USER [BRANCH]`")
            })?;
            let branch = args.next_raw().unwrap_or_else(|| "miri-sync".into());
            if args.next_raw().is_some() {
                bail!("Too many arguments for `./miri rustc-push GITHUB_USER BRANCH`");
            }
            Command::RustcPush { github_user, branch }
        }
        _ => {
            eprintln!("Unknown or missing command. Usage:\n\n{HELP}");
            std::process::exit(1);
        }
    };
    command.exec()?;
    Ok(())
}
