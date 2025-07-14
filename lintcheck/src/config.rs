use clap::{Parser, Subcommand, ValueEnum};
use std::num::NonZero;
use std::path::PathBuf;

#[allow(clippy::struct_excessive_bools)]
#[derive(Parser, Clone, Debug)]
#[command(args_conflicts_with_subcommands = true)]
pub(crate) struct LintcheckConfig {
    /// Number of threads to use (default: all unless --fix or --recursive)
    #[clap(
        long = "jobs",
        short = 'j',
        value_name = "N",
        default_value_t = 0,
        default_value_if("perf", "true", Some("1")), // Limit jobs to 1 when benchmarking
        conflicts_with("perf"),
        required = false,
        hide_default_value = true
    )]
    pub max_jobs: usize,
    /// Set the path for a crates.toml where lintcheck should read the sources from
    #[clap(
        long = "crates-toml",
        value_name = "CRATES-SOURCES-TOML-PATH",
        default_value = "lintcheck/lintcheck_crates.toml",
        hide_default_value = true,
        env = "LINTCHECK_TOML",
        hide_env = true
    )]
    pub sources_toml_path: PathBuf,
    /// File to save the clippy lint results here
    #[clap(skip = "")]
    pub lintcheck_results_path: PathBuf, // Overridden in new()
    /// Only process a single crate on the list
    #[clap(long, value_name = "CRATE")]
    pub only: Option<String>,
    /// Runs cargo clippy --fix and checks if all suggestions apply
    #[clap(long, conflicts_with("max_jobs"))]
    pub fix: bool,
    /// Apply a filter to only collect specified lints
    #[clap(long = "filter", value_name = "clippy_lint_name", use_value_delimiter = true)]
    pub lint_filter: Vec<String>,
    /// Check all Clippy lints, by default only `clippy::all` and `clippy::pedantic` are checked.
    /// Usually, it's better to use `--filter` instead
    #[clap(long, conflicts_with("lint_filter"))]
    pub all_lints: bool,
    /// Set the output format of the log file
    #[clap(long, short, default_value = "text")]
    pub format: OutputFormat,
    /// Run clippy on the dependencies of crates specified in crates-toml
    #[clap(long, conflicts_with("max_jobs"))]
    pub recursive: bool,
    /// Also produce a `perf.data` file, implies --jobs=1,
    /// the `perf.data` file can be found at
    /// `target/lintcheck/sources/<package>-<version>/perf.data`
    #[clap(long)]
    pub perf: bool,
    #[command(subcommand)]
    pub subcommand: Option<Commands>,
}

#[derive(Subcommand, Clone, Debug)]
pub(crate) enum Commands {
    /// Display a markdown diff between two lintcheck log files in JSON format
    Diff {
        old: PathBuf,
        new: PathBuf,
        /// This will limit the number of warnings that will be printed for each lint
        #[clap(long)]
        truncate: bool,
        /// Write the diff summary to a JSON file if there are any changes
        #[clap(long, value_name = "PATH")]
        write_summary: Option<PathBuf>,
    },
    /// Create a lintcheck crates TOML file containing the top N popular crates
    Popular {
        /// Output TOML file name
        output: PathBuf,
        /// Number of crate names to download
        #[clap(short, long, default_value_t = 100)]
        number: usize,
    },
}

#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OutputFormat {
    Text,
    Markdown,
    Json,
}

impl OutputFormat {
    fn file_extension(self) -> &'static str {
        match self {
            OutputFormat::Text => "txt",
            OutputFormat::Markdown => "md",
            OutputFormat::Json => "json",
        }
    }
}

impl LintcheckConfig {
    pub fn new() -> Self {
        let mut config = LintcheckConfig::parse();

        // for the path where we save the lint results, get the filename without extension (so for
        // wasd.toml, use "wasd"...)
        let filename: PathBuf = config.sources_toml_path.file_stem().unwrap().into();
        config.lintcheck_results_path = PathBuf::from(format!(
            "lintcheck-logs/{}_logs.{}",
            filename.display(),
            config.format.file_extension(),
        ));

        // look at the --threads arg, if 0 is passed, use the threads count
        if config.max_jobs == 0 {
            config.max_jobs = if config.fix || config.recursive {
                1
            } else {
                std::thread::available_parallelism().map_or(1, NonZero::get)
            };
        }

        for lint_name in &mut config.lint_filter {
            *lint_name = format!(
                "clippy::{}",
                lint_name
                    .strip_prefix("clippy::")
                    .unwrap_or(lint_name)
                    .replace('_', "-")
            );
        }

        config
    }
}
