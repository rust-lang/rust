// Inspired by Clang's clang-format-diff:
//
// https://github.com/llvm-mirror/clang/blob/master/tools/clang-format/clang-format-diff.py

#![deny(warnings)]

#[macro_use]
extern crate log;

use serde::{Deserialize, Serialize};
use serde_json as json;
use thiserror::Error;

use std::collections::HashSet;
use std::env;
use std::ffi::OsStr;
use std::io::{self, BufRead};
use std::process;

use regex::Regex;

use structopt::clap::AppSettings;
use structopt::StructOpt;

/// The default pattern of files to format.
///
/// We only want to format rust files by default.
const DEFAULT_PATTERN: &str = r".*\.rs";

#[derive(Error, Debug)]
enum FormatDiffError {
    #[error("{0}")]
    IncorrectOptions(#[from] getopts::Fail),
    #[error("{0}")]
    IncorrectFilter(#[from] regex::Error),
    #[error("{0}")]
    IoError(#[from] io::Error),
}

#[derive(StructOpt, Debug)]
#[structopt(
    name = "rustfmt-format-diff",
    setting = AppSettings::DisableVersion,
    setting = AppSettings::NextLineHelp
)]
pub struct Opts {
    /// Skip the smallest prefix containing NUMBER slashes
    #[structopt(
        short = "p",
        long = "skip-prefix",
        value_name = "NUMBER",
        default_value = "0"
    )]
    skip_prefix: u32,

    /// Custom pattern selecting file paths to reformat
    #[structopt(
        short = "f",
        long = "filter",
        value_name = "PATTERN",
        default_value = DEFAULT_PATTERN
    )]
    filter: String,
}

fn main() {
    env_logger::Builder::from_env("RUSTFMT_LOG").init();
    let opts = Opts::from_args();
    if let Err(e) = run(opts) {
        println!("{}", e);
        Opts::clap().print_help().expect("cannot write to stdout");
        process::exit(1);
    }
}

#[derive(Debug, Eq, PartialEq, Serialize, Deserialize)]
struct Range {
    file: String,
    range: [u32; 2],
}

fn run(opts: Opts) -> Result<(), FormatDiffError> {
    let (files, ranges) = scan_diff(io::stdin(), opts.skip_prefix, &opts.filter)?;
    run_rustfmt(&files, &ranges)
}

fn run_rustfmt(files: &HashSet<String>, ranges: &[Range]) -> Result<(), FormatDiffError> {
    if files.is_empty() || ranges.is_empty() {
        debug!("No files to format found");
        return Ok(());
    }

    let ranges_as_json = json::to_string(ranges).unwrap();

    debug!("Files: {:?}", files);
    debug!("Ranges: {:?}", ranges);

    let rustfmt_var = env::var_os("RUSTFMT");
    let rustfmt = match &rustfmt_var {
        Some(rustfmt) => rustfmt,
        None => OsStr::new("rustfmt"),
    };
    let exit_status = process::Command::new(rustfmt)
        .args(files)
        .arg("--file-lines")
        .arg(ranges_as_json)
        .status()?;

    if !exit_status.success() {
        return Err(FormatDiffError::IoError(io::Error::new(
            io::ErrorKind::Other,
            format!("rustfmt failed with {}", exit_status),
        )));
    }
    Ok(())
}

/// Scans a diff from `from`, and returns the set of files found, and the ranges
/// in those files.
fn scan_diff<R>(
    from: R,
    skip_prefix: u32,
    file_filter: &str,
) -> Result<(HashSet<String>, Vec<Range>), FormatDiffError>
where
    R: io::Read,
{
    let diff_pattern = format!(r"^\+\+\+\s(?:.*?/){{{}}}(\S*)", skip_prefix);
    let diff_pattern = Regex::new(&diff_pattern).unwrap();

    let lines_pattern = Regex::new(r"^@@.*\+(\d+)(,(\d+))?").unwrap();

    let file_filter = Regex::new(&format!("^{}$", file_filter))?;

    let mut current_file = None;

    let mut files = HashSet::new();
    let mut ranges = vec![];
    for line in io::BufReader::new(from).lines() {
        let line = line.unwrap();

        if let Some(captures) = diff_pattern.captures(&line) {
            current_file = Some(captures.get(1).unwrap().as_str().to_owned());
        }

        let file = match current_file {
            Some(ref f) => &**f,
            None => continue,
        };

        // FIXME(emilio): We could avoid this most of the time if needed, but
        // it's not clear it's worth it.
        if !file_filter.is_match(file) {
            continue;
        }

        let lines_captures = match lines_pattern.captures(&line) {
            Some(captures) => captures,
            None => continue,
        };

        let start_line = lines_captures
            .get(1)
            .unwrap()
            .as_str()
            .parse::<u32>()
            .unwrap();
        let line_count = match lines_captures.get(3) {
            Some(line_count) => line_count.as_str().parse::<u32>().unwrap(),
            None => 1,
        };

        if line_count == 0 {
            continue;
        }

        let end_line = start_line + line_count - 1;
        files.insert(file.to_owned());
        ranges.push(Range {
            file: file.to_owned(),
            range: [start_line, end_line],
        });
    }

    Ok((files, ranges))
}

#[test]
fn scan_simple_git_diff() {
    const DIFF: &str = include_str!("test/bindgen.diff");
    let (files, ranges) = scan_diff(DIFF.as_bytes(), 1, r".*\.rs").expect("scan_diff failed?");

    assert!(
        files.contains("src/ir/traversal.rs"),
        "Should've matched the filter"
    );

    assert!(
        !files.contains("tests/headers/anon_enum.hpp"),
        "Shouldn't have matched the filter"
    );

    assert_eq!(
        &ranges,
        &[
            Range {
                file: "src/ir/item.rs".to_owned(),
                range: [148, 158],
            },
            Range {
                file: "src/ir/item.rs".to_owned(),
                range: [160, 170],
            },
            Range {
                file: "src/ir/traversal.rs".to_owned(),
                range: [9, 16],
            },
            Range {
                file: "src/ir/traversal.rs".to_owned(),
                range: [35, 43],
            },
        ]
    );
}

#[cfg(test)]
mod cmd_line_tests {
    use super::*;

    #[test]
    fn default_options() {
        let empty: Vec<String> = vec![];
        let o = Opts::from_iter(&empty);
        assert_eq!(DEFAULT_PATTERN, o.filter);
        assert_eq!(0, o.skip_prefix);
    }

    #[test]
    fn good_options() {
        let o = Opts::from_iter(&["test", "-p", "10", "-f", r".*\.hs"]);
        assert_eq!(r".*\.hs", o.filter);
        assert_eq!(10, o.skip_prefix);
    }

    #[test]
    fn unexpected_option() {
        assert!(
            Opts::clap()
                .get_matches_from_safe(&["test", "unexpected"])
                .is_err()
        );
    }

    #[test]
    fn unexpected_flag() {
        assert!(
            Opts::clap()
                .get_matches_from_safe(&["test", "--flag"])
                .is_err()
        );
    }

    #[test]
    fn overridden_option() {
        assert!(
            Opts::clap()
                .get_matches_from_safe(&["test", "-p", "10", "-p", "20"])
                .is_err()
        );
    }

    #[test]
    fn negative_filter() {
        assert!(
            Opts::clap()
                .get_matches_from_safe(&["test", "-p", "-1"])
                .is_err()
        );
    }
}
