use std::path::PathBuf;

use clap::Parser;
use llvm_bitcode_linker::{Optimization, Session, Target};

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
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::FmtSubscriber::builder().with_max_level(tracing::Level::DEBUG).init();

    let args = Args::parse();

    let mut linker = Session::new(args.target, args.target_cpu, args.target_feature, args.output);

    linker.add_exported_symbols(args.export_symbol);

    for rlib in args.files {
        linker.add_file(rlib);
    }

    linker.lto(args.optimization, args.debug)
}
