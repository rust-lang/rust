// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(catch_panic)]

extern crate rustfmt;
extern crate diff;
extern crate regex;

use std::collections::HashMap;
use std::fs;
use std::io::{self, Read, BufRead, BufReader};
use std::thread;
use rustfmt::*;

fn get_path_string(dir_entry: io::Result<fs::DirEntry>) -> String {
    let path = dir_entry.ok().expect("Couldn't get DirEntry.").path();

    path.to_str().expect("Couldn't stringify path.").to_owned()
}

// Integration tests and idempotence tests. The files in the tests/source are
// formatted and compared to their equivalent in tests/target. The target file
// and config can be overriden by annotations in the source file. The input and
// output must match exactly.
// Files in tests/target are checked to be unaltered by rustfmt.
// FIXME(#28) would be good to check for error messages and fail on them, or at least report.
#[test]
fn system_tests() {
    // Get all files in the tests/target directory
    let files = fs::read_dir("tests/target").ok().expect("Couldn't read dir 1.");
    let files = files.chain(fs::read_dir("tests").ok().expect("Couldn't read dir 2."));
    let files = files.chain(fs::read_dir("src/bin").ok().expect("Couldn't read dir 3."));
    // turn a DirEntry into a String that represents the relative path to the file
    let files = files.map(get_path_string);
    // hack because there's no `IntoIterator` impl for `[T; N]`
    let files = files.chain(Some("src/lib.rs".to_owned()).into_iter());

    let (count, fails) = check_files(files);

    // Display results
    println!("Ran {} idempotent tests.", count);
    assert!(fails == 0, "{} idempotent tests failed", fails);

    // Get all files in the tests/source directory
    let files = fs::read_dir("tests/source").ok().expect("Couldn't read dir 4.");
    // turn a DirEntry into a String that represents the relative path to the file
    let files = files.map(get_path_string);

    let (count, fails) = check_files(files);

    // Display results
    println!("Ran {} system tests.", count);
    assert!(fails == 0, "{} system tests failed", fails);
}

// For each file, run rustfmt and collect the output.
// Returns the number of files checked and the number of failures.
fn check_files<I>(files: I) -> (u32, u32)
    where I: Iterator<Item = String>
{
    let mut count = 0;
    let mut fails = 0;

    for file_name in files.filter(|f| f.ends_with(".rs")) {
        println!("Testing '{}'...", file_name);
        match idempotent_check(file_name) {
            Ok(()) => {},
            Err(m) => {
                print_mismatches(m);
                fails += 1;
            },
        }
        count += 1;
    }

    (count, fails)
}

fn print_mismatches(result: HashMap<String, String>) {
    for (_, fmt_text) in result {
        println!("{}", fmt_text);
    }
}

// Ick, just needed to get a &'static to handle_result.
static HANDLE_RESULT: &'static Fn(HashMap<String, String>) = &handle_result;

pub fn idempotent_check(filename: String) -> Result<(), HashMap<String, String>> {
    let config = get_config(&filename);
    let args = vec!["rustfmt".to_owned(), filename];
    // this thread is not used for concurrency, but rather to workaround the issue that the passed
    // function handle needs to have static lifetime. Instead of using a global RefCell, we use
    // panic to return a result in case of failure. This has the advantage of smoothing the road to
    // multithreaded rustfmt
    thread::catch_panic(move || {
        run(args, WriteMode::Return(HANDLE_RESULT), &config);
    }).map_err(|any|
        *any.downcast().ok().expect("Downcast failed.")
    )
}

// Reads test config file from comments and loads it
fn get_config(file_name: &str) -> String {
    let config_file_name = read_significant_comment(file_name, "config")
        .map(|file_name| {
            let mut full_path = "tests/config/".to_owned();
            full_path.push_str(&file_name);
            full_path
        })
        .unwrap_or("default.toml".to_owned());

    let mut def_config_file = fs::File::open(config_file_name).ok().expect("Couldn't open config.");
    let mut def_config = String::new();
    def_config_file.read_to_string(&mut def_config).ok().expect("Couldn't read config.");

    def_config
}

fn read_significant_comment(file_name: &str, option: &str) -> Option<String> {
    let file = fs::File::open(file_name).ok().expect("Couldn't read file for comment.");
    let reader = BufReader::new(file);
    let pattern = format!("^\\s*//\\s*rustfmt-{}:\\s*(\\S+)", option);
    let regex = regex::Regex::new(&pattern).ok().expect("Failed creating pattern 1.");

    // matches exactly the lines containing significant comments or whitespace
    let line_regex = regex::Regex::new(r"(^\s*$)|(^\s*//\s*rustfmt-[:alpha:]+:\s*\S+)")
        .ok().expect("Failed creating pattern 2.");

    reader.lines()
          .map(|line| line.ok().expect("Failed getting line."))
          .take_while(|line| line_regex.is_match(&line))
          .filter_map(|line| {
              regex.captures_iter(&line).next().map(|capture| {
                  capture.at(1).expect("Couldn't unwrap capture.").to_owned()
              })
          })
          .next()
}

// Compare output to input.
fn handle_result(result: HashMap<String, String>) {
    let mut failures = HashMap::new();

    for (file_name, fmt_text) in result {
        // If file is in tests/source, compare to file with same name in tests/target
        let target_file_name = get_target(&file_name);
        let mut f = fs::File::open(&target_file_name).ok().expect("Couldn't open target.");

        let mut text = String::new();
        // TODO: speedup by running through bytes iterator
        f.read_to_string(&mut text).ok().expect("Failed reading target.");
        if fmt_text != text {
            let diff_str = make_diff(&file_name, &fmt_text, &text);
            failures.insert(file_name, diff_str);
        }
    }
    if !failures.is_empty() {
        panic!(failures);
    }
}

// Map source file paths to their target paths.
fn get_target(file_name: &str) -> String {
    if file_name.starts_with("tests/source/") {
        let target = read_significant_comment(file_name, "target");
        let base = target.unwrap_or(file_name.trim_left_matches("tests/source/").to_owned());

        let mut target_file = "tests/target/".to_owned();
        target_file.push_str(&base);

        target_file
    } else {
        file_name.to_owned()
    }
}

// Produces a diff string between the expected output and actual output of
// rustfmt on a given file
fn make_diff(file_name: &str, expected: &str, actual: &str) -> String {
    let mut line_number = 1;
    let mut prev_both = true;
    let mut text = String::new();

    for result in diff::lines(expected, actual) {
        match result {
            diff::Result::Left(str) => {
                if prev_both {
                    text.push_str(&format!("Mismatch @ {}:{}\n", file_name, line_number));
                }
                text.push_str(&format!("-{}⏎\n", str));
                prev_both = false;
            }
            diff::Result::Right(str) => {
                if prev_both {
                    text.push_str(&format!("Mismatch @ {}:{}\n", file_name, line_number));
                }
                text.push_str(&format!("+{}⏎\n", str));
                prev_both = false;
                line_number += 1;
            }
            diff::Result::Both(..) => {
                line_number += 1;
                prev_both = true;
            }
        }
    }

    text
}
