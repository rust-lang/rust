use std::path::PathBuf;
use std::str::FromStr;

use anyhow::Result;
use clap::{Parser, ValueEnum};
use tidy::features::Version;

use crate::err::DumpError;

#[derive(Debug, Parser)]
#[command(version, about)]
pub struct Cli {
    /// Path to `library/` directory. Use this flag to read features from the standard library.
    #[arg(long)]
    pub library_path: Option<PathBuf>,
    /// Path to `compiler/` directory. Use this flag to read language features.
    #[arg(long)]
    pub compiler_path: Option<PathBuf>,
    /// Which file to write to. If none, writes to stdout.
    #[arg(long)]
    pub output_path: Option<PathBuf>,

    /// What file format to write to. Text is the human-readable option.
    #[arg(long)]
    #[arg(default_value = "json")]
    pub format: Format,

    /// Which features to show first. Only has effect when `format = text`.
    /// Features two features with equal versions are ordered by issue number.
    /// Features with no version are considered old.
    #[arg(long)]
    #[arg(default_value = "newest")]
    pub sort_by: SortBy,

    /// How to filter unstable features.
    #[arg(long)]
    #[arg(default_value = "allow")]
    #[arg(conflicts_with_all = ["accepted", "removed"])]
    pub unstable: Tristate,

    /// How to filter accepted (stable) features.
    #[arg(long)]
    #[arg(default_value = "allow")]
    #[arg(conflicts_with_all = ["removed", "unstable"])]
    pub accepted: Tristate,

    /// How to filter removed features.
    #[arg(long)]
    #[arg(default_value = "allow")]
    #[arg(conflicts_with_all = ["accepted", "unstable"])]
    pub removed: Tristate,

    /// How to filter issues with(out) a tracking issue.
    #[arg(long)]
    #[arg(default_value = "allow")]
    pub tracking_issue: Tristate,

    /// How to filter issues with(out) `since` version.
    #[arg(long)]
    #[arg(default_value = "allow")]
    #[arg(conflicts_with_all(["first_version", "last_version"]))]
    pub since: Tristate,

    /// Only show features introduced after or in this version (semver triple)
    /// Features without known version are filtered out using this flag.
    /// Features notated with `Current Version` are considered newer than any concrete semver.
    #[arg(long)]
    #[arg(value_parser = Version::from_str)]
    pub first_version: Option<Version>,

    /// Only show features introduced before or in this version (semver triple).
    /// Features without known version are filtered out using this flag.
    /// Features notated with `Current Version` are considered newer than any concrete semver.
    #[arg(long)]
    #[arg(value_parser = Version::from_str)]
    pub last_version: Option<Version>,
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
pub enum Tristate {
    /// Only show these features.
    Require,
    /// Has no effect.
    Allow,
    /// Do not show these features.
    Deny,
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
pub enum Format {
    /// Formats into JSON.
    /// Contains two objects "lang_features_status" and "lib_features_status",
    /// each containing strings (feature names) mapping to tidy::features::Feature objects.
    JSON,
    /// formats each feature into a line like
    ///
    /// > [SOURCE] NAME is STATUS since VERSION <LINK TO ISSUE>: DESCRIPTION
    ///
    /// Leaving out the unknown parts.
    Text,
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
pub enum SortBy {
    Oldest,
    Newest,
}

pub fn parse() -> Result<Cli> {
    let cli = Cli::parse();

    if cli.compiler_path == None && cli.library_path == None {
        return Err(DumpError::NoSources.into());
    }

    if let Some(first_version) = cli.first_version
        && let Some(last_version) = cli.last_version
        && first_version > last_version
    {
        return Err(DumpError::NoVersions.into());
    }

    Ok(cli)
}
