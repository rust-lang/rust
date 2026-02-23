use std::num::NonZeroUsize;
use std::path::PathBuf;

use clap::{Arg, ArgAction, ArgMatches, Command, value_parser};

#[cfg(test)]
mod tests;

#[derive(Debug, Clone)]
pub struct TidyArgParser {
    pub root_path: PathBuf,
    pub cargo: PathBuf,
    pub output_directory: PathBuf,
    pub concurrency: NonZeroUsize,
    pub npm: PathBuf,
    pub verbose: bool,
    pub bless: bool,
    pub extra_checks: Option<Vec<String>>,
    pub pos_args: Vec<String>,
}

impl TidyArgParser {
    fn command() -> Command {
        Command::new("rust-tidy")
            .arg(
                Arg::new("root_path")
                    .help("path of the root directory")
                    .long("root-path")
                    .required(true)
                    .value_parser(value_parser!(PathBuf)),
            )
            .arg(
                Arg::new("cargo")
                    .help("path of cargo")
                    .long("cargo-path")
                    .required(true)
                    .value_parser(value_parser!(PathBuf)),
            )
            .arg(
                Arg::new("output_directory")
                    .help("path of output directory")
                    .long("output-dir")
                    .required(true)
                    .value_parser(value_parser!(PathBuf)),
            )
            .arg(
                Arg::new("concurrency")
                    .help("number of threads working concurrently")
                    .long("concurrency")
                    .required(true)
                    .value_parser(value_parser!(NonZeroUsize)),
            )
            .arg(
                Arg::new("npm")
                    .help("path of npm")
                    .long("npm-path")
                    .required(true)
                    .value_parser(value_parser!(PathBuf)),
            )
            .arg(Arg::new("verbose").help("verbose").long("verbose").action(ArgAction::SetTrue))
            .arg(Arg::new("bless").help("target files are modified").long("bless").action(ArgAction::SetTrue))
            .arg(
                Arg::new("extra_checks")
                    .help("extra checks")
                    .long("extra-checks")
                    .value_delimiter(',')
                    .action(ArgAction::Append),
            )
            .arg(Arg::new("pos_args").help("for extra checks. you can specify configs and target files for external check tools").action(ArgAction::Append).last(true))
    }

    fn build(matches: ArgMatches) -> Self {
        let mut tidy_flags = Self {
            root_path: matches.get_one::<PathBuf>("root_path").unwrap().clone(),
            cargo: matches.get_one::<PathBuf>("cargo").unwrap().clone(),
            output_directory: matches.get_one::<PathBuf>("output_directory").unwrap().clone(),
            concurrency: *matches.get_one::<NonZeroUsize>("concurrency").unwrap(),
            npm: matches.get_one::<PathBuf>("npm").unwrap().clone(),
            verbose: *matches.get_one::<bool>("verbose").unwrap(),
            bless: *matches.get_one::<bool>("bless").unwrap(),
            extra_checks: None,
            pos_args: vec![],
        };

        if let Some(extra_checks) = matches.get_many::<String>("extra_checks") {
            tidy_flags.extra_checks = Some(extra_checks.map(|s| s.to_string()).collect::<Vec<_>>());
        }

        tidy_flags.pos_args = matches
            .get_many::<String>("pos_args")
            .unwrap_or_default()
            .map(|v| v.to_string())
            .collect::<Vec<_>>();

        tidy_flags
    }

    pub fn parse() -> Self {
        let matches = Self::command().get_matches();
        Self::build(matches)
    }
}
