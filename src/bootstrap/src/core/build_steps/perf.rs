use std::fmt::{Display, Formatter};
use std::process::Command;

use crate::core::build_steps::compile::{Std, Sysroot};
use crate::core::build_steps::tool::RustcPerf;
use crate::core::builder::Builder;
use crate::core::config::DebuginfoLevel;

/// Performs profiling or benchmarking with [`rustc-perf`](https://github.com/rust-lang/rustc-perf)
/// using a locally built compiler.
#[derive(Debug, Clone, clap::Parser)]
pub struct PerfArgs {
    #[clap(subcommand)]
    cmd: PerfCommand,

    #[clap(flatten)]
    opts: SharedOpts,
}

impl Default for PerfArgs {
    fn default() -> Self {
        Self { cmd: PerfCommand::Eprintln, opts: SharedOpts::default() }
    }
}

#[derive(Debug, Clone, clap::Parser)]
enum PerfCommand {
    /// Run `profile_local eprintln`.
    /// This executes the compiler on the given benchmarks and stores its stderr output.
    Eprintln,
}

#[derive(Debug, Default, Clone, clap::Parser)]
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

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
#[value(rename_all = "PascalCase")]
enum Profile {
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
enum Scenario {
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
        compiler: builder.compiler(0, builder.config.build),
        target: builder.config.build,
    });

    if builder.build.config.rust_debuginfo_level_rustc == DebuginfoLevel::None {
        builder.info(r#"WARNING: You are compiling rustc without debuginfo, this will make profiling less useful.
Consider setting `rust.debuginfo-level = 1` in `config.toml`."#);
    }

    let compiler = builder.compiler(builder.top_stage, builder.config.build);
    builder.ensure(Std::new(compiler, builder.config.build));
    let sysroot = builder.ensure(Sysroot::new(compiler));
    let rustc = sysroot.join("bin/rustc");

    let rustc_perf_dir = builder.build.tempdir().join("rustc-perf");

    let mut cmd = Command::new(collector);
    match args.cmd {
        PerfCommand::Eprintln => {
            cmd.arg("profile_local").arg("eprintln");
            cmd.arg("--out-dir").arg(rustc_perf_dir.join("results"));
        }
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
    cmd.arg(&rustc);

    builder.info(&format!("Running `rustc-perf` using `{}`", rustc.display()));

    // We need to set the working directory to `src/tools/perf`, so that it can find the directory
    // with compile-time benchmarks.
    let cmd = cmd.current_dir(builder.src.join("src/tools/rustc-perf"));
    builder.run(cmd);

    builder.info(&format!("You can find the results at `{}`", rustc_perf_dir.display()));
}
