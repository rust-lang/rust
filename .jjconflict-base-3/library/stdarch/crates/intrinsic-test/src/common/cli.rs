use itertools::Itertools;
use std::path::PathBuf;

/// Intrinsic test tool
#[derive(clap::Parser)]
#[command(
    name = "Intrinsic test tool",
    about = "Generates Rust and C programs for intrinsics and compares the output"
)]
pub struct Cli {
    /// The input file containing the intrinsics
    pub input: PathBuf,

    /// Filename for a list of intrinsics to skip (one per line)
    #[arg(long)]
    pub skip: Vec<PathBuf>,

    /// Pass a target the test suite
    #[arg(long)]
    pub target: String,

    /// Percentage of intrinsics to test (used to limit testing to keep CI times manageable)
    #[arg(long, default_value_t = 100u8)]
    pub sample_percentage: u8,

    /// Argument style of the C compiler
    #[arg(long)]
    pub cc_arg_style: CcArgStyle,
}

#[derive(Copy, Clone, clap::ValueEnum)]
pub enum CcArgStyle {
    Gcc,
    Clang,
}

pub struct ProcessedCli {
    pub filename: PathBuf,
    pub target: String,
    pub skip: Vec<String>,
    pub sample_percentage: u8,
    pub cc_arg_style: CcArgStyle,
}

impl ProcessedCli {
    pub fn new(cli_options: Cli) -> Self {
        let filename = cli_options.input;
        let target = cli_options.target;
        let sample_percentage = cli_options.sample_percentage;

        let skip = cli_options
            .skip
            .iter()
            .flat_map(|filename| {
                std::fs::read_to_string(&filename)
                    .expect("Failed to open file")
                    .lines()
                    .map(|line| line.trim().to_owned())
                    .filter(|line| !line.contains('#'))
                    .collect_vec()
            })
            .collect_vec();

        Self {
            target,
            skip,
            filename,
            sample_percentage,
            cc_arg_style: cli_options.cc_arg_style,
        }
    }
}
