use std::path::PathBuf;

use anyhow::anyhow;
use clap::{ArgAction, Parser};
use llvm_bitcode_linker::{Optimization, Session, Target};
use tracing::level_filters::LevelFilter;

#[derive(Debug, Parser)]
/// Linker for embedded code without any system dependencies
pub struct Args {
    /// Input files - objects, archives and static libraries.
    ///
    /// An archive can be, but not required to be, a Rust rlib.
    files: Vec<PathBuf>,

    /// A symbol that should be exported
    #[arg(long)]
    export_symbol: Vec<String>,

    /// Input files directory
    #[arg(short = 'L')]
    input_dir: Vec<PathBuf>,

    /// Target triple for which the code is compiled
    #[arg(long)]
    target: Target,

    /// The target cpu
    #[arg(long)]
    target_cpu: Option<String>,

    /// The target features
    #[arg(long)]
    target_feature: Option<String>,

    /// Write output to the filename
    #[arg(short, long)]
    output: PathBuf,

    // Enable link time optimization
    #[arg(long)]
    lto: bool,

    /// Emit debug information
    #[arg(long)]
    debug: bool,

    /// The optimization level
    #[arg(short = 'O', value_enum, default_value = "0")]
    optimization: Optimization,

    /// Increase linker diagnostic verbosity (-v = info, -vv = debug)
    #[arg(short = 'v', long = "verbose", action = ArgAction::Count)]
    verbose: u8,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let max_tracing_level = match args.verbose {
        0 => LevelFilter::OFF,
        1 => LevelFilter::INFO,
        _ => LevelFilter::TRACE,
    };

    tracing_subscriber::FmtSubscriber::builder()
        .with_max_level(max_tracing_level)
        .with_target(false)
        .without_time()
        .with_level(false)
        .with_ansi(false)
        .init();

    let mut linker = Session::new(args.target, args.target_cpu, args.target_feature, args.output);

    linker.add_exported_symbols(args.export_symbol);

    for rlib in args.files {
        linker.add_file(rlib);
    }

    let hint = if max_tracing_level < LevelFilter::ERROR {
        "Pass `-v` to llvm-bitcode-linker for additional diagnostic output."
    } else {
        ""
    };

    linker.lto(args.optimization, args.debug).map_err(|err| anyhow!("{err}\n{hint}"))
}
