#![allow(clippy::needless_question_mark)]

mod commands;
mod util;

use std::env;
use std::ffi::OsString;

use anyhow::{anyhow, bail, Result};

#[derive(Clone, Debug)]
pub enum Command {
    /// Installs the miri driver and cargo-miri.
    /// Sets up the rpath such that the installed binary should work in any
    /// working directory. Note that the binaries are placed in the `miri` toolchain
    /// sysroot, to prevent conflicts with other toolchains.
    Install {
        /// Flags that are passed through to `cargo install`.
        flags: Vec<OsString>,
    },
    /// Just build miri.
    Build {
        /// Flags that are passed through to `cargo build`.
        flags: Vec<OsString>,
    },
    /// Just check miri.
    Check {
        /// Flags that are passed through to `cargo check`.
        flags: Vec<OsString>,
    },
    /// Build miri, set up a sysroot and then run the test suite.
    Test {
        bless: bool,
        /// Flags that are passed through to `cargo test`.
        flags: Vec<OsString>,
    },
    /// Build miri, set up a sysroot and then run the driver with the given <flags>.
    /// (Also respects MIRIFLAGS environment variable.)
    Run {
        dep: bool,
        /// Flags that are passed through to `miri`.
        flags: Vec<OsString>,
    },
    /// Format all sources and tests.
    Fmt {
        /// Flags that are passed through to `rustfmt`.
        flags: Vec<OsString>,
    },
    /// Runs clippy on all sources.
    Clippy {
        /// Flags that are passed through to `cargo clippy`.
        flags: Vec<OsString>,
    },
    /// Runs just `cargo <flags>` with the Miri-specific environment variables.
    /// Mainly meant to be invoked by rust-analyzer.
    Cargo { flags: Vec<OsString> },
    /// Runs <command> over and over again with different seeds for Miri. The MIRIFLAGS
    /// variable is set to its original value appended with ` -Zmiri-seed=$SEED` for
    /// many different seeds.
    ManySeeds { command: Vec<OsString> },
    /// Runs the benchmarks from bench-cargo-miri in hyperfine. hyperfine needs to be installed.
    Bench {
        /// List of benchmarks to run. By default all benchmarks are run.
        benches: Vec<OsString>,
    },
    /// Update and activate the rustup toolchain 'miri' to the commit given in the
    /// `rust-version` file.
    /// `rustup-toolchain-install-master` must be installed for this to work. Any extra
    /// flags are passed to `rustup-toolchain-install-master`.
    Toolchain { flags: Vec<OsString> },
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

./miri test [--bless] <flags>:
Build miri, set up a sysroot and then run the test suite. <flags> are passed
to the final `cargo test` invocation.

./miri run [--dep] <flags>:
Build miri, set up a sysroot and then run the driver with the given <flags>.
(Also respects MIRIFLAGS environment variable.)

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

./miri many-seeds <command>:
Runs <command> over and over again with different seeds for Miri. The MIRIFLAGS
variable is set to its original value appended with ` -Zmiri-seed=$SEED` for
many different seeds. The MIRI_SEEDS variable controls how many seeds are being
tried; MIRI_SEED_START controls the first seed to try.

./miri bench <benches>:
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

./miri rustc-push <github user> <branch>:
Push Miri changes back to the rustc repo. This will pull a copy of the rustc
history into the Miri repo, unless you set the RUSTC_GIT env var to an existing
clone of the rustc repo.

  ENVIRONMENT VARIABLES

MIRI_SYSROOT:
If already set, the "sysroot setup" step is skipped.

CARGO_EXTRA_FLAGS:
Pass extra flags to all cargo invocations. (Ignored by `./miri cargo`.)"#;

fn main() -> Result<()> {
    // We are hand-rolling our own argument parser, since `clap` can't express what we need
    // (https://github.com/clap-rs/clap/issues/5055).
    let mut args = env::args_os().peekable();
    args.next().unwrap(); // skip program name
    let command = match args.next().and_then(|s| s.into_string().ok()).as_deref() {
        Some("build") => Command::Build { flags: args.collect() },
        Some("check") => Command::Check { flags: args.collect() },
        Some("test") => {
            let bless = args.peek().is_some_and(|a| a.to_str() == Some("--bless"));
            if bless {
                // Consume the flag.
                args.next().unwrap();
            }
            Command::Test { bless, flags: args.collect() }
        }
        Some("run") => {
            let dep = args.peek().is_some_and(|a| a.to_str() == Some("--dep"));
            if dep {
                // Consume the flag.
                args.next().unwrap();
            }
            Command::Run { dep, flags: args.collect() }
        }
        Some("fmt") => Command::Fmt { flags: args.collect() },
        Some("clippy") => Command::Clippy { flags: args.collect() },
        Some("cargo") => Command::Cargo { flags: args.collect() },
        Some("install") => Command::Install { flags: args.collect() },
        Some("many-seeds") => Command::ManySeeds { command: args.collect() },
        Some("bench") => Command::Bench { benches: args.collect() },
        Some("toolchain") => Command::Toolchain { flags: args.collect() },
        Some("rustc-pull") => {
            let commit = args.next().map(|a| a.to_string_lossy().into_owned());
            if args.next().is_some() {
                bail!("Too many arguments for `./miri rustc-pull`");
            }
            Command::RustcPull { commit }
        }
        Some("rustc-push") => {
            let github_user = args
                .next()
                .ok_or_else(|| {
                    anyhow!("Missing first argument for `./miri rustc-push GITHUB_USER BRANCH`")
                })?
                .to_string_lossy()
                .into_owned();
            let branch = args
                .next()
                .ok_or_else(|| {
                    anyhow!("Missing second argument for `./miri rustc-push GITHUB_USER BRANCH`")
                })?
                .to_string_lossy()
                .into_owned();
            if args.next().is_some() {
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
