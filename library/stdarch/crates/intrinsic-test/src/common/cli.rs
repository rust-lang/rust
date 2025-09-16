use itertools::Itertools;
use std::path::PathBuf;

#[derive(Debug, PartialEq)]
pub enum Language {
    Rust,
    C,
}

pub enum FailureReason {
    RunC(String),
    RunRust(String),
    Difference(String, String, String),
}

/// Intrinsic test tool
#[derive(clap::Parser)]
#[command(
    name = "Intrinsic test tool",
    about = "Generates Rust and C programs for intrinsics and compares the output"
)]
pub struct Cli {
    /// The input file containing the intrinsics
    pub input: PathBuf,

    /// The rust toolchain to use for building the rust code
    #[arg(long)]
    pub toolchain: Option<String>,

    /// The C++ compiler to use for compiling the c++ code
    #[arg(long, default_value_t = String::from("clang++"))]
    pub cppcompiler: String,

    /// Run the C programs under emulation with this command
    #[arg(long)]
    pub runner: Option<String>,

    /// Filename for a list of intrinsics to skip (one per line)
    #[arg(long)]
    pub skip: Option<PathBuf>,

    /// Regenerate test programs, but don't build or run them
    #[arg(long)]
    pub generate_only: bool,

    /// Pass a target the test suite
    #[arg(long, default_value_t = String::from("armv7-unknown-linux-gnueabihf"))]
    pub target: String,

    /// Set the linker
    #[arg(long)]
    pub linker: Option<String>,

    /// Set the sysroot for the C++ compiler
    #[arg(long)]
    pub cxx_toolchain_dir: Option<String>,
}

pub struct ProcessedCli {
    pub filename: PathBuf,
    pub toolchain: Option<String>,
    pub cpp_compiler: Option<String>,
    pub runner: String,
    pub target: String,
    pub linker: Option<String>,
    pub cxx_toolchain_dir: Option<String>,
    pub skip: Vec<String>,
}

impl ProcessedCli {
    pub fn new(cli_options: Cli) -> Self {
        let filename = cli_options.input;
        let runner = cli_options.runner.unwrap_or_default();
        let target = cli_options.target;
        let linker = cli_options.linker;
        let cxx_toolchain_dir = cli_options.cxx_toolchain_dir;

        let skip = if let Some(filename) = cli_options.skip {
            let data = std::fs::read_to_string(&filename).expect("Failed to open file");
            data.lines()
                .map(str::trim)
                .filter(|s| !s.contains('#'))
                .map(String::from)
                .collect_vec()
        } else {
            Default::default()
        };

        let (toolchain, cpp_compiler) = if cli_options.generate_only {
            (None, None)
        } else {
            (
                Some(
                    cli_options
                        .toolchain
                        .map_or_else(String::new, |t| format!("+{t}")),
                ),
                Some(cli_options.cppcompiler),
            )
        };

        Self {
            toolchain,
            cpp_compiler,
            runner,
            target,
            linker,
            cxx_toolchain_dir,
            skip,
            filename,
        }
    }
}
