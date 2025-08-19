#![allow(clippy::needless_question_mark, rustc::internal)]

mod commands;
mod coverage;
mod util;

use anyhow::{Result, bail};
use clap::{Parser, Subcommand};

#[derive(Clone, Debug, Subcommand)]
pub enum Command {
    /// Installs the miri driver and cargo-miri to the sysroot of the active toolchain.
    ///
    /// Sets up the rpath such that the installed binary should work in any
    /// working directory.
    Install {
        /// Pass features to cargo invocations on the "miri" crate in the root. This option does
        /// **not** apply to other crates, so e.g. these features won't be used on "cargo-miri".
        #[arg(long, value_delimiter = ',', action = clap::ArgAction::Append)]
        features: Vec<String>,
        /// Flags that are passed through to `cargo install`.
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        flags: Vec<String>,
    },
    /// Build Miri.
    Build {
        /// Pass features to cargo invocations on the "miri" crate in the root. This option does
        /// **not** apply to other crates, so e.g. these features won't be used on "cargo-miri".
        #[arg(long, value_delimiter = ',', action = clap::ArgAction::Append)]
        features: Vec<String>,
        /// Flags that are passed through to `cargo build`.
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        flags: Vec<String>,
    },
    /// Check Miri.
    Check {
        /// Pass features to cargo invocations on the "miri" crate in the root. This option does
        /// **not** apply to other crates, so e.g. these features won't be used on "cargo-miri".
        #[arg(long, value_delimiter = ',', action = clap::ArgAction::Append)]
        features: Vec<String>,
        /// Flags that are passed through to `cargo check`.
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        flags: Vec<String>,
    },
    /// Check Miri with Clippy.
    Clippy {
        /// Pass features to cargo invocations on the "miri" crate in the root. This option does
        /// **not** apply to other crates, so e.g. these features won't be used on "cargo-miri".
        #[arg(long, value_delimiter = ',', action = clap::ArgAction::Append)]
        features: Vec<String>,
        /// Flags that are passed through to `cargo clippy`.
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        flags: Vec<String>,
    },
    /// Run the Miri test suite.
    Test {
        /// Update stdout/stderr reference files.
        #[arg(long)]
        bless: bool,
        /// The cross-interpretation target.
        #[arg(long)]
        target: Option<String>,
        /// Produce coverage report.
        #[arg(long)]
        coverage: bool,
        /// Pass features to cargo invocations on the "miri" crate in the root. This option does
        /// **not** apply to other crates, so e.g. these features won't be used on "cargo-miri".
        #[arg(long, value_delimiter = ',', action = clap::ArgAction::Append)]
        features: Vec<String>,
        /// Flags that are passed through to the test harness.
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        flags: Vec<String>,
    },
    /// Run the Miri driver.
    ///
    /// Also respects MIRIFLAGS environment variable.
    Run {
        /// Build the program with the dependencies declared in `tests/deps/Cargo.toml`.
        #[arg(long)]
        dep: bool,
        /// Show build progress.
        #[arg(long, short)]
        verbose: bool,
        /// The cross-interpretation target.
        #[arg(long)]
        target: Option<String>,
        /// The Rust edition.
        #[arg(long)]
        edition: Option<String>,
        /// Pass features to cargo invocations on the "miri" crate in the root. This option does
        /// **not** apply to other crates, so e.g. these features won't be used on "cargo-miri".
        #[arg(long, value_delimiter = ',', action = clap::ArgAction::Append)]
        features: Vec<String>,
        /// Flags that are passed through to `miri`.
        ///
        /// The flags set in `MIRIFLAGS` are added in front of these flags.
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        flags: Vec<String>,
    },
    /// Build documentation.
    Doc {
        /// Pass features to cargo invocations on the "miri" crate in the root. This option does
        /// **not** apply to other crates, so e.g. these features won't be used on "cargo-miri".
        #[arg(long, value_delimiter = ',', action = clap::ArgAction::Append)]
        features: Vec<String>,
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
    /// Runs the benchmarks from bench-cargo-miri in hyperfine.
    ///
    /// hyperfine needs to be installed.
    Bench {
        #[arg(long)]
        target: Option<String>,
        /// When `true`, skip the `./miri install` step.
        #[arg(long)]
        no_install: bool,
        /// Store the benchmark result in the given file, so it can be used
        /// as the baseline for a future run.
        #[arg(long)]
        save_baseline: Option<String>,
        /// Load previous stored benchmark results as baseline, and print an analysis of how the
        /// current run compares.
        #[arg(long)]
        load_baseline: Option<String>,
        /// List of benchmarks to run (default: run all benchmarks).
        benches: Vec<String>,
    },
    /// Update and activate the rustup toolchain 'miri'.
    ///
    /// The `rust-version` file is used to determine the commit that will be intsalled.
    /// `rustup-toolchain-install-master` must be installed for this to work.
    Toolchain {
        /// Flags that are passed through to `rustup-toolchain-install-master`.
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        flags: Vec<String>,
    },
    /// Squash the commits of the current feature branch into one.
    Squash,
}

impl Command {
    fn add_remainder(&mut self, remainder: Vec<String>) -> Result<()> {
        if remainder.is_empty() {
            return Ok(());
        }

        match self {
            Self::Install { flags, .. }
            | Self::Build { flags, .. }
            | Self::Check { flags, .. }
            | Self::Doc { flags, .. }
            | Self::Fmt { flags }
            | Self::Toolchain { flags }
            | Self::Clippy { flags, .. }
            | Self::Run { flags, .. }
            | Self::Test { flags, .. } => {
                flags.extend(remainder);
                Ok(())
            }
            Self::Bench { .. } | Self::Squash => bail!("unexpected \"--\" found in arguments"),
        }
    }
}

#[derive(Parser)]
#[command(after_help = "Environment variables:
  MIRI_SYSROOT: If already set, the \"sysroot setup\" step is skipped
  CARGO_EXTRA_FLAGS: Pass extra flags to all cargo invocations")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

fn main() -> Result<()> {
    // If we are invoked as the git sequence editor, jump to that logic.
    if !std::env::var_os("MIRI_SCRIPT_IS_GIT_SEQUENCE_EDITOR").unwrap_or_default().is_empty() {
        return Command::squash_sequence_editor();
    }

    // Split the arguments into the part before the `--` and the part after.
    // The `--` itself ends up in the second part.
    let miri_args: Vec<_> = std::env::args().take_while(|x| *x != "--").collect();
    let remainder: Vec<_> = std::env::args().skip_while(|x| *x != "--").collect();

    let args = Cli::parse_from(miri_args);
    let mut command = args.command;
    command.add_remainder(remainder)?;
    command.exec()?;
    Ok(())
}
