use std::path::PathBuf;
use std::str::FromStr;

use anyhow::Result;
use clap::{Parser, ValueEnum};
use tidy::features::Version;

use crate::err::{DumpError, SemverFlag};

#[derive(Debug, Parser)]
#[command(version, about)]
struct Cli {
    /// Path to `library/` directory. Use this flag to read features from the standard library.
    #[arg(long)]
    library_path: Option<PathBuf>,
    /// Path to `compiler/` directory. Use this flag to read language features.
    #[arg(long)]
    compiler_path: Option<PathBuf>,
    /// Which file to write to. If none, writes to stdout.
    #[arg(long)]
    output_path: Option<PathBuf>,

    /// What file format to write to. Text is the human-readable option.
    #[arg(long)]
    #[arg(default_value = "json")]
    format: Format,

    /// Which features to show first. Only has effect when `format = text`.
    /// Features two features with equal versions are ordered by issue number.
    /// Features with no version are considered old.
    #[arg(long)]
    #[arg(default_value = "newest")]
    sort_by: SortBy,

    /// How to filter unstable features.
    #[arg(long)]
    #[arg(default_value = "allow")]
    unstable: Tristate,

    /// How to filter accepted (stable) features.
    #[arg(long)]
    #[arg(default_value = "allow")]
    accepted: Tristate,

    /// How to filter removed features.
    #[arg(long)]
    #[arg(default_value = "allow")]
    removed: Tristate,

    /// How to filter issues with(out) a tracking issue.
    #[arg(long)]
    #[arg(default_value = "allow")]
    tracking_issue: Tristate,

    /// Only show features introduced before this version (semver triple).
    /// Features without known version are considered oldest.
    /// Features notated with `Current Version` are considered newer than any concrete semver.
    #[arg(long)]
    before: Option<String>,

    /// Only show features introduced after or in this version (semver triple)
    #[arg(long)]
    since: Option<String>,
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
pub enum Tristate {
    /// Only show these features
    Require,
    /// Has no effect
    Allow,
    /// Do not show these features
    Deny,
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
pub enum Format {
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

// Not everything can be parsed by clap, so there's a second struct for properly parsed arguments.
pub struct Args {
    pub library_path: Option<PathBuf>,
    pub compiler_path: Option<PathBuf>,
    pub output_path: Option<PathBuf>,

    pub format: Format,
    pub sort_by: SortBy,

    pub unstable: Tristate,
    pub accepted: Tristate,
    pub removed: Tristate,
    pub tracking_issue: Tristate,

    pub before: Option<Version>,
    pub since: Option<Version>,
}

impl Args {
    pub fn parse() -> Result<Self> {
        let cli = Cli::parse();

        if cli.compiler_path == None && cli.library_path == None {
            return Err(DumpError::NoSources.into());
        }

        let before = match &cli.before {
            Some(string) => {
                let result = Version::from_str(string.as_str());
                // tidy's error is opaque and does not implement Display, so we cannot propagate it into anyhow.
                let version = result.map_err(|_| DumpError::BadSemver {
                    cause: string.clone(),
                    flag: SemverFlag::Before,
                })?;
                Some(version)
            }
            None => None,
        };

        let since = match &cli.since {
            Some(string) => {
                let result = Version::from_str(string.as_str());
                // tidy's error is opaque and does not implement Display, so we cannot propagate it into anyhow.
                let version = result.map_err(|_| DumpError::BadSemver {
                    cause: string.clone(),
                    flag: SemverFlag::Since,
                })?;
                Some(version)
            }
            None => None,
        };

        if let Some(since) = since
            && let Some(before) = before
            && since >= before
        {
            return Err(DumpError::NoVersions.into());
        }

        let require_count = [cli.unstable, cli.accepted, cli.removed]
            .iter()
            .filter(|&&x| Tristate::Require == x)
            .count();
        if require_count > 1 {
            return Err(DumpError::StatusConflict.into());
        }

        let args = Args {
            library_path: cli.library_path,
            compiler_path: cli.compiler_path,
            output_path: cli.output_path,
            format: cli.format,
            sort_by: cli.sort_by,
            unstable: cli.unstable,
            accepted: cli.accepted,
            removed: cli.removed,
            tracking_issue: cli.tracking_issue,
            before,
            since,
        };
        Ok(args)
    }
}
