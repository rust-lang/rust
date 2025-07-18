use std::env::consts::EXE_EXTENSION;
use std::fmt::{Display, Formatter};

use crate::core::build_steps::compile::Sysroot;
use crate::core::build_steps::tool::{RustcPerf, Rustdoc};
use crate::core::builder::Builder;
use crate::core::config::DebuginfoLevel;
use crate::utils::exec::{BootstrapCommand, command};

#[derive(Debug, Clone, clap::Parser)]
pub struct PerfArgs {
    #[clap(subcommand)]
    cmd: PerfCommand,
}

#[derive(Debug, Clone, clap::Parser)]
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
    /// Run compile benchmarks with a locally built compiler.
    Benchmark {
        /// Identifier to associate benchmark results with
        #[clap(name = "benchmark-id")]
        id: String,

        #[clap(flatten)]
        opts: SharedOpts,
    },
    /// Compare the results of two previously executed benchmark runs.
    Compare {
        /// The name of the base artifact to be compared.
        base: String,

        /// The name of the modified artifact to be compared.
        modified: String,
    },
}

impl PerfCommand {
    fn shared_opts(&self) -> Option<&SharedOpts> {
        match self {
            PerfCommand::Eprintln { opts, .. }
            | PerfCommand::Samply { opts, .. }
            | PerfCommand::Cachegrind { opts, .. }
            | PerfCommand::Benchmark { opts, .. } => Some(opts),
            PerfCommand::Compare { .. } => None,
        }
    }
}

#[derive(Debug, Clone, clap::Parser)]
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

#[derive(Clone, Copy, Debug, PartialEq, clap::ValueEnum)]
#[value(rename_all = "PascalCase")]
pub enum Profile {
    Check,
    Debug,
    Doc,
    Opt,
    Clippy,
}

impl Display for Profile {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Profile::Check => "Check",
            Profile::Debug => "Debug",
            Profile::Doc => "Doc",
            Profile::Opt => "Opt",
            Profile::Clippy => "Clippy",
        };
        f.write_str(name)
    }
}

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
#[value(rename_all = "PascalCase")]
pub enum Scenario {
    Full,
    IncrFull,
    IncrUnchanged,
    IncrPatched,
}

impl Display for Scenario {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Scenario::Full => "Full",
            Scenario::IncrFull => "IncrFull",
            Scenario::IncrUnchanged => "IncrUnchanged",
            Scenario::IncrPatched => "IncrPatched",
        };
        f.write_str(name)
    }
}

/// Performs profiling using `rustc-perf` on a built version of the compiler.
pub fn perf(builder: &Builder<'_>, args: &PerfArgs) {
    let collector = builder.ensure(RustcPerf {
        compiler: builder.compiler(0, builder.config.host_target),
        target: builder.config.host_target,
    });

    let is_profiling = match &args.cmd {
        PerfCommand::Eprintln { .. }
        | PerfCommand::Samply { .. }
        | PerfCommand::Cachegrind { .. } => true,
        PerfCommand::Benchmark { .. } | PerfCommand::Compare { .. } => false,
    };
    if is_profiling && builder.build.config.rust_debuginfo_level_rustc == DebuginfoLevel::None {
        builder.info(r#"WARNING: You are compiling rustc without debuginfo, this will make profiling less useful.
Consider setting `rust.debuginfo-level = 1` in `bootstrap.toml`."#);
    }

    let compiler = builder.compiler(builder.top_stage, builder.config.host_target);
    builder.std(compiler, builder.config.host_target);

    if let Some(opts) = args.cmd.shared_opts()
        && opts.profiles.contains(&Profile::Doc)
    {
        builder.ensure(Rustdoc { compiler });
    }

    let sysroot = builder.ensure(Sysroot::new(compiler));
    let mut rustc = sysroot.clone();
    rustc.push("bin");
    rustc.push("rustc");
    rustc.set_extension(EXE_EXTENSION);

    let rustc_perf_dir = builder.build.tempdir().join("rustc-perf");
    let results_dir = rustc_perf_dir.join("results");
    builder.create_dir(&results_dir);

    let mut cmd = command(collector.tool_path);

    // We need to set the working directory to `src/tools/rustc-perf`, so that it can find the directory
    // with compile-time benchmarks.
    cmd.current_dir(builder.src.join("src/tools/rustc-perf"));

    let db_path = results_dir.join("results.db");

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

            cmd.arg("--out-dir").arg(&results_dir);
            cmd.arg(rustc);

            apply_shared_opts(&mut cmd, opts);
            cmd.run(builder);

            println!("You can find the results at `{}`", &results_dir.display());
        }
        PerfCommand::Benchmark { id, opts } => {
            cmd.arg("bench_local");
            cmd.arg("--db").arg(&db_path);
            cmd.arg("--id").arg(id);
            cmd.arg(rustc);

            apply_shared_opts(&mut cmd, opts);
            cmd.run(builder);
        }
        PerfCommand::Compare { base, modified } => {
            cmd.arg("bench_cmp");
            cmd.arg("--db").arg(&db_path);
            cmd.arg(base).arg(modified);

            cmd.run(builder);
        }
    }
}

fn apply_shared_opts(cmd: &mut BootstrapCommand, opts: &SharedOpts) {
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
