// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern crate rustfmt;
extern crate diff;
extern crate regex;
extern crate term;

use std::collections::HashMap;
use std::fs;
use std::io::{self, Read, BufRead, BufReader};
use std::path::Path;

use rustfmt::*;
use rustfmt::config::{Config, ReportTactic};
use rustfmt::rustfmt_diff::*;

static DIFF_CONTEXT_SIZE: usize = 3;

fn get_path_string(dir_entry: io::Result<fs::DirEntry>) -> String {
    let path = dir_entry.ok().expect("Couldn't get DirEntry.").path();

    path.to_str().expect("Couldn't stringify path.").to_owned()
}

// Integration tests. The files in the tests/source are formatted and compared
// to their equivalent in tests/target. The target file and config can be
// overriden by annotations in the source file. The input and output must match
// exactly.
// FIXME(#28) would be good to check for error messages and fail on them, or at
// least report.
#[test]
fn system_tests() {
    // Get all files in the tests/source directory.
    let files = fs::read_dir("tests/source").ok().expect("Couldn't read source dir.");
    // Turn a DirEntry into a String that represents the relative path to the
    // file.
    let files = files.map(get_path_string);
    let (_reports, count, fails) = check_files(files, WriteMode::Return);

    // Display results.
    println!("Ran {} system tests.", count);
    assert!(fails == 0, "{} system tests failed", fails);
}

// Do the same for tests/coverage-source directory
// the only difference is the coverage mode
#[test]
fn coverage_tests() {
    let files = fs::read_dir("tests/coverage-source").ok().expect("Couldn't read source dir.");
    let files = files.map(get_path_string);
    let (_reports, count, fails) = check_files(files, WriteMode::Coverage);

    println!("Ran {} tests in coverage mode.", count);
    assert!(fails == 0, "{} tests failed", fails);
}

// Idempotence tests. Files in tests/target are checked to be unaltered by
// rustfmt.
#[test]
fn idempotence_tests() {
    // Get all files in the tests/target directory.
    let files = fs::read_dir("tests/target")
                    .ok()
                    .expect("Couldn't read target dir.")
                    .map(get_path_string);
    let (_reports, count, fails) = check_files(files, WriteMode::Return);

    // Display results.
    println!("Ran {} idempotent tests.", count);
    assert!(fails == 0, "{} idempotent tests failed", fails);
}

// Run rustfmt on itself. This operation must be idempotent. We also check that
// no warnings are emitted.
#[test]
fn self_tests() {
    let files = fs::read_dir("src/bin")
                    .ok()
                    .expect("Couldn't read src dir.")
                    .chain(fs::read_dir("tests").ok().expect("Couldn't read tests dir."))
                    .map(get_path_string);
    // Hack because there's no `IntoIterator` impl for `[T; N]`.
    let files = files.chain(Some("src/lib.rs".to_owned()).into_iter());

    let (reports, count, fails) = check_files(files, WriteMode::Return);
    let mut warnings = 0;

    // Display results.
    println!("Ran {} self tests.", count);
    assert!(fails == 0, "{} self tests failed", fails);

    for format_report in reports {
        println!("{}", format_report);
        warnings += format_report.warning_count();
    }

    assert!(warnings == 0,
            "Rustfmt's code generated {} warnings",
            warnings);
}

// For each file, run rustfmt and collect the output.
// Returns the number of files checked and the number of failures.
fn check_files<I>(files: I, write_mode: WriteMode) -> (Vec<FormatReport>, u32, u32)
    where I: Iterator<Item = String>
{
    let mut count = 0;
    let mut fails = 0;
    let mut reports = vec![];

    for file_name in files.filter(|f| f.ends_with(".rs")) {
        println!("Testing '{}'...", file_name);

        match idempotent_check(file_name, write_mode) {
            Ok(report) => reports.push(report),
            Err(msg) => {
                print_mismatches(msg);
                fails += 1;
            }
        }

        count += 1;
    }

    (reports, count, fails)
}

fn print_mismatches(result: HashMap<String, Vec<Mismatch>>) {
    let mut t = term::stdout().unwrap();

    for (file_name, diff) in result {
        print_diff(diff,
                   |line_num| format!("\nMismatch at {}:{}:", file_name, line_num));
    }

    assert!(t.reset().unwrap());
}

pub fn idempotent_check(filename: String,
                        write_mode: WriteMode)
                        -> Result<FormatReport, HashMap<String, Vec<Mismatch>>> {
    let sig_comments = read_significant_comments(&filename);
    let mut config = get_config(sig_comments.get("config").map(|x| &(*x)[..]));

    for (key, val) in &sig_comments {
        if key != "target" && key != "config" {
            config.override_value(key, val);
        }
    }

    // Don't generate warnings for to-do items.
    config.report_todo = ReportTactic::Never;

    let mut file_map = format(Path::new(&filename), &config, write_mode);
    let format_report = fmt_lines(&mut file_map, &config);

    // Won't panic, as we're not doing any IO.
    let write_result = filemap::write_all_files(&file_map, WriteMode::Return, &config).unwrap();
    let target = sig_comments.get("target").map(|x| &(*x)[..]);

    handle_result(write_result, target, write_mode).map(|_| format_report)
}

// Reads test config file from comments and reads its contents.
fn get_config(config_file: Option<&str>) -> Config {
    let config_file_name = match config_file {
        None => return Default::default(),
        Some(file_name) => {
            let mut full_path = "tests/config/".to_owned();
            full_path.push_str(&file_name);
            full_path
        }
    };

    let mut def_config_file = fs::File::open(config_file_name)
                                  .ok()
                                  .expect("Couldn't open config.");
    let mut def_config = String::new();
    def_config_file.read_to_string(&mut def_config).ok().expect("Couldn't read config.");

    Config::from_toml(&def_config)
}

// Reads significant comments of the form: // rustfmt-key: value
// into a hash map.
fn read_significant_comments(file_name: &str) -> HashMap<String, String> {
    let file = fs::File::open(file_name)
                   .ok()
                   .expect(&format!("Couldn't read file {}.", file_name));
    let reader = BufReader::new(file);
    let pattern = r"^\s*//\s*rustfmt-([^:]+):\s*(\S+)";
    let regex = regex::Regex::new(&pattern).ok().expect("Failed creating pattern 1.");

    // Matches lines containing significant comments or whitespace.
    let line_regex = regex::Regex::new(r"(^\s*$)|(^\s*//\s*rustfmt-[^:]+:\s*\S+)")
                         .ok()
                         .expect("Failed creating pattern 2.");

    reader.lines()
          .map(|line| line.ok().expect("Failed getting line."))
          .take_while(|line| line_regex.is_match(&line))
          .filter_map(|line| {
              regex.captures_iter(&line).next().map(|capture| {
                  (capture.at(1).expect("Couldn't unwrap capture.").to_owned(),
                   capture.at(2).expect("Couldn't unwrap capture.").to_owned())
              })
          })
          .collect()
}

// Compare output to input.
// TODO: needs a better name, more explanation.
fn handle_result(result: HashMap<String, String>,
                 target: Option<&str>,
                 write_mode: WriteMode)
                 -> Result<(), HashMap<String, Vec<Mismatch>>> {
    let mut failures = HashMap::new();

    for (file_name, fmt_text) in result {
        // If file is in tests/source, compare to file with same name in tests/target.
        let target = get_target(&file_name, target, write_mode);
        let mut f = fs::File::open(&target).ok().expect("Couldn't open target.");

        let mut text = String::new();
        f.read_to_string(&mut text).ok().expect("Failed reading target.");

        if fmt_text != text {
            let diff = make_diff(&text, &fmt_text, DIFF_CONTEXT_SIZE);
            failures.insert(file_name, diff);
        }
    }

    if failures.is_empty() {
        Ok(())
    } else {
        Err(failures)
    }
}

// Map source file paths to their target paths.
fn get_target(file_name: &str, target: Option<&str>, write_mode: WriteMode) -> String {
    let file_path = Path::new(file_name);
    let (source_path_prefix, target_path_prefix) = match write_mode {
        WriteMode::Coverage => {
            (Path::new("tests/coverage-source/"),
             "tests/coverage-target/")
        }
        _ => (Path::new("tests/source/"), "tests/target/"),
    };

    if file_path.starts_with(source_path_prefix) {
        let mut components = file_path.components();
        // Can't skip(2) as the resulting iterator can't as_path()
        components.next();
        components.next();

        let new_target = match components.as_path().to_str() {
            Some(string) => string,
            None => file_name,
        };
        let base = target.unwrap_or(new_target);

        format!("{}{}", target_path_prefix, base)
    } else {
        file_name.to_owned()
    }
}
