use clap::Parser;
use std::path::PathBuf;

#[derive(Clone, Debug, Parser)]
pub(crate) struct LintcheckConfig {
    /// Number of threads to use, 0 automatic choice
    #[clap(long = "jobs", short = 'j', value_name = "N", default_value_t = 1)]
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
    /// Apply a filter to only collect specified lints, this also overrides `allow` attributes
    #[clap(long = "filter", value_name = "clippy_lint_name", use_value_delimiter = true)]
    pub lint_filter: Vec<String>,
    /// Change the reports table to use markdown links
    #[clap(long)]
    pub markdown: bool,
    /// Run clippy on the dependencies of crates specified in crates-toml
    #[clap(long, conflicts_with("max_jobs"))]
    pub recursive: bool,
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
            if config.markdown { "md" } else { "txt" }
        ));

        // look at the --threads arg, if 0 is passed, use the threads count
        if config.max_jobs == 0 {
            // automatic choice
            config.max_jobs = std::thread::available_parallelism().map_or(1, |n| n.get());
        };

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
