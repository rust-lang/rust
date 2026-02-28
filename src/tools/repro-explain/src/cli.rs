use camino::Utf8PathBuf;
use clap::{Parser, Subcommand, ValueEnum};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, ValueEnum, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum ConfirmLevel {
    None,
    Cheap,
    Standard,
    Full,
}

impl Default for ConfirmLevel {
    fn default() -> Self {
        Self::Standard
    }
}

#[derive(Debug, Parser)]
#[command(name = "repro-explain", version, about = "Explain Rust build reproducibility diffs")]
pub struct Cli {
    /// Working directory root (default: .repro)
    #[arg(long, default_value = ".repro", global = true)]
    pub work_dir: Utf8PathBuf,

    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Run a build command twice and complete capture+diff+explain+report.
    RunTwice {
        /// Confirmation effort level.
        #[arg(long, value_enum, default_value_t = ConfirmLevel::Standard)]
        confirm: ConfirmLevel,
        /// Replay job count (used by confirmation replays).
        #[arg(long, default_value_t = 1)]
        jobs: usize,
        /// Keep target directory between capture runs.
        #[arg(long)]
        keep_target: bool,
        /// Explicit path to diffoscope.
        #[arg(long)]
        diffoscope: Option<Utf8PathBuf>,
        /// Disable diffoscope usage.
        #[arg(long)]
        no_diffoscope: bool,
        /// Capture all environment variables.
        #[arg(long)]
        capture_all_env: bool,
        /// Enable same-source-dir replay for path leak confirmation.
        #[arg(long)]
        same_source_replay: bool,
        /// Enable nightly -Z binary-dep-depinfo.
        #[arg(long)]
        binary_dep_depinfo: bool,
        /// Build command to execute (use `--` delimiter).
        #[arg(required = true, trailing_var_arg = true)]
        command: Vec<String>,
    },

    /// Execute one capture run.
    Capture {
        /// Run ID (for example A, B).
        #[arg(long)]
        run_id: String,
        /// Keep target directory between capture runs.
        #[arg(long)]
        keep_target: bool,
        /// Capture all environment variables.
        #[arg(long)]
        capture_all_env: bool,
        /// Enable nightly -Z binary-dep-depinfo.
        #[arg(long)]
        binary_dep_depinfo: bool,
        /// Build command to execute (use `--` delimiter).
        #[arg(required = true, trailing_var_arg = true)]
        command: Vec<String>,
    },

    /// Compare two capture runs and generate diff manifest.
    Diff {
        #[arg(long)]
        left: Utf8PathBuf,
        #[arg(long)]
        right: Utf8PathBuf,
    },

    /// Explain an existing analysis directory.
    Explain {
        #[arg(long)]
        analysis: Utf8PathBuf,
        #[arg(long)]
        artifact: Option<String>,
        #[arg(long, value_enum, default_value_t = ConfirmLevel::Standard)]
        confirm: ConfirmLevel,
        #[arg(long, default_value_t = 1)]
        jobs: usize,
        #[arg(long)]
        same_source_replay: bool,
        #[arg(long)]
        diffoscope: Option<Utf8PathBuf>,
        #[arg(long)]
        no_diffoscope: bool,
    },

    /// Regenerate HTML/JSON report from analysis directory.
    Report {
        #[arg(long)]
        analysis: Utf8PathBuf,
    },

    /// Internal rustc wrapper mode.
    #[command(name = "__wrap-rustc", hide = true)]
    WrapRustc {
        #[arg(required = true, trailing_var_arg = true)]
        argv: Vec<String>,
    },

    /// Internal rustdoc wrapper mode.
    #[command(name = "__wrap-rustdoc", hide = true)]
    WrapRustdoc {
        #[arg(trailing_var_arg = true)]
        argv: Vec<String>,
    },
}

#[derive(Debug, Clone)]
pub struct CaptureFlags {
    pub keep_target: bool,
    pub capture_all_env: bool,
    pub binary_dep_depinfo: bool,
}

#[derive(Debug, Clone)]
pub struct ExplainFlags {
    pub artifact_glob: Option<String>,
    pub confirm: ConfirmLevel,
    pub jobs: usize,
    pub same_source_replay: bool,
    pub diffoscope: Option<Utf8PathBuf>,
    pub no_diffoscope: bool,
}
