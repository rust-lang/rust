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
    pub skip: Option<PathBuf>,

    /// Pass a target the test suite
    #[arg(long)]
    pub target: String,

    #[arg(long, default_value_t = 100u8)]
    pub sample_percentage: u8,
}

pub struct ProcessedCli {
    pub filename: PathBuf,
    pub target: String,
    pub skip: Vec<String>,
    pub sample_percentage: u8,
}

impl ProcessedCli {
    pub fn new(cli_options: Cli) -> Self {
        let filename = cli_options.input;
        let target = cli_options.target;
        let sample_percentage = cli_options.sample_percentage;

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

        Self {
            target,
            skip,
            filename,
            sample_percentage,
        }
    }
}
