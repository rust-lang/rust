use std::fs::create_dir_all;
use std::path::PathBuf;
use std::process::Command;

use clap::Parser;

use crate::config::{Profile, Scenario};

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
    ctx: BuildContext,
}

#[derive(Debug, clap::Parser)]
enum PerfCommand {
    /// Run `profile_local eprintln`.
    /// This executes the compiler on the given benchmarks and stores its stderr output.
    Eprintln {
        #[clap(flatten)]
        opts: SharedOpts,
    },
    /// Run `profile_local samply`
    /// This executes the compiler on the given benchmarks and profiles it with `samply`.
    /// You need to install `samply`, e.g. using `cargo install samply`.
    Samply {
        #[clap(flatten)]
        opts: SharedOpts,
    },
    /// Run `profile_local cachegrind`.
    /// This executes the compiler on the given benchmarks under `Cachegrind`.
    Cachegrind {
        #[clap(flatten)]
        opts: SharedOpts,
    },
    Benchmark {
        /// Identifier to associate benchmark results with
        id: String,

        #[clap(flatten)]
        opts: SharedOpts,
    },
    Compare {
        /// The name of the base artifact to be compared.
        base: String,

        /// The name of the modified artifact to be compared.
        modified: String,
    },
}

#[derive(Debug, clap::Parser)]
struct SharedOpts {
    /// Select the benchmarks that you want to run (separated by commas).
    /// If unspecified, all benchmarks will be executed.
    #[clap(long, global = true, value_delimiter = ',')]
    include: Vec<String>,

    /// Select the benchmarks matching a prefix in this comma-separated list that you don't want to run.
    #[clap(long, global = true, value_delimiter = ',')]
    exclude: Vec<String>,

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
    let db_path = args.ctx.results_dir.join("results.db");

    match &args.cmd {
        PerfCommand::Eprintln { opts }
        | PerfCommand::Samply { opts }
        | PerfCommand::Cachegrind { opts } => {
            cmd.arg("profile_local");
            cmd.arg(match &args.cmd {
                PerfCommand::Eprintln { .. } => "eprintln",
                PerfCommand::Samply { .. } => "samply",
                PerfCommand::Cachegrind { .. } => "cachegrind",
                _ => unreachable!(),
            });

            cmd.arg("--out-dir").arg(&args.ctx.results_dir);

            apply_shared_opts(&mut cmd, opts);
            execute_benchmark(&mut cmd, &args.ctx.compiler);

            println!("You can find the results at `{}`", args.ctx.results_dir.display());
        }
        PerfCommand::Benchmark { id, opts } => {
            cmd.arg("bench_local");
            cmd.arg("--db").arg(&db_path);
            cmd.arg("--id").arg(id);

            apply_shared_opts(&mut cmd, opts);
            create_dir_all(&args.ctx.results_dir).unwrap();
            execute_benchmark(&mut cmd, &args.ctx.compiler);
        }
        PerfCommand::Compare { base, modified } => {
            cmd.arg("bench_cmp");
            cmd.arg("--db").arg(&db_path);
            cmd.arg(base).arg(modified);

            create_dir_all(&args.ctx.results_dir).unwrap();
            cmd.status().expect("error while running rustc-perf bench_cmp");
        }
    }
}

fn apply_shared_opts(cmd: &mut Command, opts: &SharedOpts) {
    if !opts.include.is_empty() {
        cmd.arg("--include").arg(opts.include.join(","));
    }
    if !opts.exclude.is_empty() {
        cmd.arg("--exclude").arg(opts.exclude.join(","));
    }
    if !opts.profiles.is_empty() {
        cmd.arg("--profiles")
            .arg(opts.profiles.iter().map(|p| p.to_string()).collect::<Vec<_>>().join(","));
    }
    if !opts.scenarios.is_empty() {
        cmd.arg("--scenarios")
            .arg(opts.scenarios.iter().map(|p| p.to_string()).collect::<Vec<_>>().join(","));
    }
}

fn execute_benchmark(cmd: &mut Command, compiler: &Path) {
    cmd.arg(compiler);
    println!("Running `rustc-perf` using `{}`", compiler.display());

    const MANIFEST_DIR: &str = env!("CARGO_MANIFEST_DIR");

    let rustc_perf_dir = PathBuf::from(MANIFEST_DIR).join("../rustc-perf");

    // We need to set the working directory to `src/tools/perf`, so that it can find the directory
    // with compile-time benchmarks.
    let cmd = cmd.current_dir(rustc_perf_dir);
    cmd.status().expect("error while running rustc-perf collector");
}
