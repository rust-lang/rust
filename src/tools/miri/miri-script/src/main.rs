#![allow(clippy::needless_question_mark)]

mod commands;
mod coverage;
mod util;

use std::ops::Range;

use anyhow::{Context, Result, anyhow};
use clap::{Parser, Subcommand};

/// Parses a seed range
///
/// This function is used for the `--many-seeds` flag. It expects the range in the form
/// `<from>..<to>`. `<from>` is inclusive, `<to>` is exclusive. `<from>` can be omitted,
/// in which case it is assumed to be `0`.
fn parse_range(val: &str) -> anyhow::Result<Range<u32>> {
    let (from, to) = val
        .split_once("..")
        .ok_or_else(|| anyhow!("invalid format for `--many-seeds`: expected `from..to`"))?;
    let from: u32 = if from.is_empty() {
        0
    } else {
        from.parse().context("invalid `from` in `--many-seeds=from..to")?
    };
    let to: u32 = to.parse().context("invalid `to` in `--many-seeds=from..to")?;
    Ok(from..to)
}

#[derive(Clone, Debug, Subcommand)]
pub enum Command {
    /// Installs the miri driver and cargo-miri.
    /// Sets up the rpath such that the installed binary should work in any
    /// working directory. Note that the binaries are placed in the `miri` toolchain
    /// sysroot, to prevent conflicts with other toolchains.
    Install {
        /// Flags that are passed through to `cargo install`.
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        flags: Vec<String>,
    },
    /// Just build miri.
    Build {
        /// Flags that are passed through to `cargo build`.
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        flags: Vec<String>,
    },
    /// Just check miri.
    Check {
        /// Flags that are passed through to `cargo check`.
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        flags: Vec<String>,
    },
    /// Build miri, set up a sysroot and then run the test suite.
    Test {
        #[arg(long)]
        bless: bool,
        /// The cross-interpretation target.
        /// If none then the host is the target.
        #[arg(long)]
        target: Option<String>,
        /// Produce coverage report if set.
        #[arg(long)]
        coverage: bool,
        /// Flags that are passed through to the test harness.
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        flags: Vec<String>,
    },
    /// Build miri, set up a sysroot and then run the driver with the given <flags>.
    /// (Also respects MIRIFLAGS environment variable.)
    Run {
        #[arg(long)]
        dep: bool,
        #[arg(long, short)]
        verbose: bool,
        #[arg(long, value_parser = parse_range)]
        many_seeds: Option<Range<u32>>,
        #[arg(long)]
        target: Option<String>,
        #[arg(long)]
        edition: Option<String>,
        /// Flags that are passed through to `miri`.
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        flags: Vec<String>,
    },
    /// Build documentation
    Doc {
        /// Flags that are passed through to `cargo doc`.
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        flags: Vec<String>,
    },
    /// Format all sources and tests.
    Fmt {
        /// Flags that are passed through to `rustfmt`.
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        flags: Vec<String>,
    },
    /// Runs clippy on all sources.
    Clippy {
        /// Flags that are passed through to `cargo clippy`.
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        flags: Vec<String>,
    },
    /// Runs the benchmarks from bench-cargo-miri in hyperfine. hyperfine needs to be installed.
    Bench {
        #[arg(long)]
        target: Option<String>,
        /// When `true`, skip the `./miri install` step.
        no_install: bool,
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

impl Command {
    fn add_remainder(&mut self, remainder: Vec<String>) -> Result<()> {
        if remainder.is_empty() {
            return Ok(());
        }

        match self {
            Self::Install { flags }
            | Self::Build { flags }
            | Self::Check { flags }
            | Self::Doc { flags }
            | Self::Fmt { flags }
            | Self::Toolchain { flags }
            | Self::Clippy { flags }
            | Self::Run { flags, .. }
            | Self::Test { flags, .. } => {
                flags.extend(remainder);
                Ok(())
            }
            Self::Bench { .. } | Self::RustcPull { .. } | Self::RustcPush { .. } =>
                Err(anyhow::Error::msg("unexpected \"--\" found in arguments")),
        }
    }
}

#[derive(Parser)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

fn main() -> Result<()> {
    let miri_args: Vec<_> = std::env::args().take_while(|x| *x != "--").collect();
    let remainder: Vec<_> = std::env::args().skip_while(|x| *x != "--").collect();

    let args = Cli::parse_from(miri_args);
    let mut command = args.command;
    command.add_remainder(remainder)?;
    command.exec()?;
    Ok(())
}
