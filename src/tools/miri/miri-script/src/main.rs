mod commands;
mod util;

use std::ffi::OsString;

use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser, Clone, Debug)]
#[command(author, about, long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand, Clone, Debug)]
pub enum Command {
    /// Installs the miri driver and cargo-miri.
    /// Sets up the rpath such that the installed binary should work in any
    /// working directory. Note that the binaries are placed in the `miri` toolchain
    /// sysroot, to prevent conflicts with other toolchains.
    Install {
        /// Flags that are passed through to `cargo install`.
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        flags: Vec<OsString>,
    },
    /// Just build miri.
    Build {
        /// Flags that are passed through to `cargo build`.
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        flags: Vec<OsString>,
    },
    /// Just check miri.
    Check {
        /// Flags that are passed through to `cargo check`.
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        flags: Vec<OsString>,
    },
    /// Build miri, set up a sysroot and then run the test suite.
    Test {
        #[arg(long, default_value_t = false)]
        bless: bool,
        /// Flags that are passed through to `cargo test`.
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        flags: Vec<OsString>,
    },
    /// Build miri, set up a sysroot and then run the driver with the given <flags>.
    /// (Also respects MIRIFLAGS environment variable.)
    Run {
        #[arg(long, default_value_t = false)]
        dep: bool,
        /// Flags that are passed through to `miri`
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        flags: Vec<OsString>,
    },
    /// Format all sources and tests.
    Fmt {
        /// Flags that are passed through to `rustfmt`.
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        flags: Vec<OsString>,
    },
    /// Runs clippy on all sources.
    Clippy {
        /// Flags that are passed through to `cargo clippy`.
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        flags: Vec<OsString>,
    },
    /// Runs just `cargo <flags>` with the Miri-specific environment variables.
    /// Mainly meant to be invoked by rust-analyzer.
    Cargo {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        flags: Vec<OsString>,
    },
    /// Runs <command> over and over again with different seeds for Miri. The MIRIFLAGS
    /// variable is set to its original value appended with ` -Zmiri-seed=$SEED` for
    /// many different seeds.
    ManySeeds {
        /// Starting seed.
        #[arg(long, env = "MIRI_SEED_START", default_value_t = 0)]
        seed_start: u64,
        /// Amount of seeds to try.
        #[arg(long, env = "MIRI_SEEDS", default_value_t = 256)]
        seeds: u64,
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        command: Vec<OsString>,
    },
    /// Runs the benchmarks from bench-cargo-miri in hyperfine. hyperfine needs to be installed.
    Bench {
        /// List of benchmarks to run. By default all benchmarks are run.
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        benches: Vec<OsString>,
    },
    /// Update and activate the rustup toolchain 'miri' to the commit given in the
    /// `rust-version` file.
    /// `rustup-toolchain-install-master` must be installed for this to work. Any extra
    /// flags are passed to `rustup-toolchain-install-master`.
    Toolchain {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        flags: Vec<OsString>,
    },
    /// Pull and merge Miri changes from the rustc repo. Defaults to fetching the latest
    /// rustc commit. The fetched commit is stored in the `rust-version` file, so the
    /// next `./miri toolchain` will install the rustc that just got pulled.
    RustcPull { commit: Option<String> },
    /// Push Miri changes back to the rustc repo. This will pull a copy of the rustc
    /// history into the Miri repo, unless you set the RUSTC_GIT env var to an existing
    /// clone of the rustc repo.
    RustcPush {
        #[arg(long, env = "RUSTC_GIT")]
        rustc_git: Option<String>,
        github_user: String,
        branch: String,
    },
}

fn main() -> Result<()> {
    let args = Cli::parse();
    args.command.exec()?;
    Ok(())
}
