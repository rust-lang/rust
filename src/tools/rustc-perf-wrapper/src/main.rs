use crate::config::{Profile, Scenario};
use clap::Parser;
use std::path::PathBuf;
use std::process::Command;

mod config;

/// Performs profiling or benchmarking with [`rustc-perf`](https://github.com/rust-lang/rustc-perf)
/// using a locally built compiler.
#[derive(Debug, clap::Parser)]
// Hide arguments from BuildContext in the default usage string.
// Clap does not seem to have a way of disabling the usage of these arguments.
#[clap(override_usage = "rustc-perf-wrapper [OPTIONS] <COMMAND>")]
pub struct Args {
    #[clap(subcommand)]
    cmd: PerfCommand,

    #[clap(flatten)]
    opts: SharedOpts,

    #[clap(flatten)]
    ctx: BuildContext,
}

#[derive(Debug, clap::Parser)]
enum PerfCommand {
    /// Run `profile_local eprintln`.
    /// This executes the compiler on the given benchmarks and stores its stderr output.
    Eprintln,
    /// Run `profile_local samply`
    /// This executes the compiler on the given benchmarks and profiles it with `samply`.
    /// You need to install `samply`, e.g. using `cargo install samply`.
    Samply,
    /// Run `profile_local cachegrind`.
    /// This executes the compiler on the given benchmarks under `Cachegrind`.
    Cachegrind,
}

impl PerfCommand {
    fn is_profiling(&self) -> bool {
        match self {
            PerfCommand::Eprintln | PerfCommand::Samply | PerfCommand::Cachegrind => true,
        }
    }
}

#[derive(Debug, clap::Parser)]
struct SharedOpts {
    /// Select the benchmarks that you want to run (separated by commas).
    /// If unspecified, all benchmarks will be executed.
    #[clap(long, global = true, value_delimiter = ',')]
    include: Vec<String>,
    /// Select the scenarios that should be benchmarked.
    #[clap(
        long,
        global = true,
        value_delimiter = ',',
        default_value = "Full,IncrFull,IncrUnchanged,IncrPatched"
    )]
    scenarios: Vec<Scenario>,
    /// Select the profiles that should be benchmarked.
    #[clap(long, global = true, value_delimiter = ',', default_value = "Check,Debug,Opt")]
    profiles: Vec<Profile>,
}

/// These arguments are mostly designed to be passed from bootstrap, not by users
/// directly.
#[derive(Debug, clap::Parser)]
struct BuildContext {
    /// Compiler binary that will be benchmarked/profiled.
    #[clap(long, hide = true, env = "RUSTC_REAL")]
    compiler: PathBuf,
    /// rustc-perf collector binary that will be used for running benchmarks/profilers.
    #[clap(long, hide = true, env = "PERF_COLLECTOR")]
    collector: PathBuf,
    /// Directory where to store results.
    #[clap(long, hide = true, env = "PERF_RESULT_DIR")]
    results_dir: PathBuf,
}

fn main() {
    let args = Args::parse();
    run(args);
}

fn run(args: Args) {
    let mut cmd = Command::new(args.ctx.collector);
    match &args.cmd {
        PerfCommand::Eprintln => {
            cmd.arg("profile_local").arg("eprintln");
        }
        PerfCommand::Samply => {
            cmd.arg("profile_local").arg("samply");
        }
        PerfCommand::Cachegrind => {
            cmd.arg("profile_local").arg("cachegrind");
        }
    }
    if args.cmd.is_profiling() {
        cmd.arg("--out-dir").arg(&args.ctx.results_dir);
    }

    if !args.opts.include.is_empty() {
        cmd.arg("--include").arg(args.opts.include.join(","));
    }
    if !args.opts.profiles.is_empty() {
        cmd.arg("--profiles")
            .arg(args.opts.profiles.iter().map(|p| p.to_string()).collect::<Vec<_>>().join(","));
    }
    if !args.opts.scenarios.is_empty() {
        cmd.arg("--scenarios")
            .arg(args.opts.scenarios.iter().map(|p| p.to_string()).collect::<Vec<_>>().join(","));
    }
    cmd.arg(&args.ctx.compiler);

    println!("Running `rustc-perf` using `{}`", args.ctx.compiler.display());

    const MANIFEST_DIR: &str = env!("CARGO_MANIFEST_DIR");

    let rustc_perf_dir = PathBuf::from(MANIFEST_DIR).join("../rustc-perf");

    // We need to set the working directory to `src/tools/perf`, so that it can find the directory
    // with compile-time benchmarks.
    let cmd = cmd.current_dir(rustc_perf_dir);
    cmd.status().expect("error while running rustc-perf collector");

    if args.cmd.is_profiling() {
        println!("You can find the results at `{}`", args.ctx.results_dir.display());
    }
}
