// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Inspired by Clang's clang-format-diff:
//
// https://github.com/llvm-mirror/clang/blob/master/tools/clang-format/clang-format-diff.py

#![deny(warnings)]

extern crate env_logger;
extern crate getopts;
#[macro_use]
extern crate log;
extern crate regex;
#[macro_use]
extern crate serde_derive;
extern crate serde_json as json;

use std::{env, fmt, process};
use std::collections::HashSet;
use std::error::Error;
use std::io::{self, BufRead};

use regex::Regex;

/// The default pattern of files to format.
///
/// We only want to format rust files by default.
const DEFAULT_PATTERN: &str = r".*\.rs";

#[derive(Debug)]
enum FormatDiffError {
    IncorrectOptions(getopts::Fail),
    IncorrectFilter(regex::Error),
    IoError(io::Error),
}

impl fmt::Display for FormatDiffError {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        fmt::Display::fmt(self.cause().unwrap(), f)
    }
}

impl Error for FormatDiffError {
    fn description(&self) -> &str {
        self.cause().unwrap().description()
    }

    fn cause(&self) -> Option<&Error> {
        Some(match *self {
            FormatDiffError::IoError(ref e) => e,
            FormatDiffError::IncorrectFilter(ref e) => e,
            FormatDiffError::IncorrectOptions(ref e) => e,
        })
    }
}

impl From<getopts::Fail> for FormatDiffError {
    fn from(fail: getopts::Fail) -> Self {
        FormatDiffError::IncorrectOptions(fail)
    }
}

impl From<regex::Error> for FormatDiffError {
    fn from(err: regex::Error) -> Self {
        FormatDiffError::IncorrectFilter(err)
    }
}

impl From<io::Error> for FormatDiffError {
    fn from(fail: io::Error) -> Self {
        FormatDiffError::IoError(fail)
    }
}

fn main() {
    let _ = env_logger::init();

    let mut opts = getopts::Options::new();
    opts.optflag("h", "help", "show this message");
    opts.optopt(
        "p",
        "skip-prefix",
        "skip the smallest prefix containing NUMBER slashes",
        "NUMBER",
    );
    opts.optopt(
        "f",
        "filter",
        "custom pattern selecting file paths to reformat",
        "PATTERN",
    );

    if let Err(e) = run(&opts) {
        println!("{}", opts.usage(e.description()));
        process::exit(1);
    }
}

#[derive(Debug, Eq, PartialEq, Serialize, Deserialize)]
struct Range {
    file: String,
    range: [u32; 2],
}

fn run(opts: &getopts::Options) -> Result<(), FormatDiffError> {
    let matches = opts.parse(env::args().skip(1))?;

    if matches.opt_present("h") {
        println!("{}", opts.usage("usage: "));
        return Ok(());
    }

    let filter = matches
        .opt_str("f")
        .unwrap_or_else(|| DEFAULT_PATTERN.to_owned());

    let skip_prefix = matches
        .opt_str("p")
        .and_then(|p| p.parse::<u32>().ok())
        .unwrap_or(0);

    let (files, ranges) = scan_diff(io::stdin(), skip_prefix, &filter)?;

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

    let exit_status = process::Command::new("rustfmt")
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

        // TODO(emilio): We could avoid this most of the time if needed, but
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
    const DIFF: &'static str = include_str!("test/bindgen.diff");
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
            }
        ]
    );
}
