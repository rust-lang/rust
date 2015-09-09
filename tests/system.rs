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
extern crate term;

use std::collections::HashMap;
use std::fs;
use std::io::{self, Read, BufRead, BufReader};
use std::thread;
use rustfmt::*;
use rustfmt::config::Config;
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
    // Get all files in the tests/source directory
    let files = fs::read_dir("tests/source").ok().expect("Couldn't read source dir.");
    // turn a DirEntry into a String that represents the relative path to the file
    let files = files.map(get_path_string);

    let (count, fails) = check_files(files);

    // Display results
    println!("Ran {} system tests.", count);
    assert!(fails == 0, "{} system tests failed", fails);
}

// Idempotence tests. Files in tests/target are checked to be unaltered by
// rustfmt.
#[test]
fn idempotence_tests() {
    // Get all files in the tests/target directory
    let files = fs::read_dir("tests/target").ok().expect("Couldn't read target dir.");
    let files = files.chain(fs::read_dir("tests").ok().expect("Couldn't read tests dir."));
    let files = files.chain(fs::read_dir("src/bin").ok().expect("Couldn't read src dir."));
    // turn a DirEntry into a String that represents the relative path to the file
    let files = files.map(get_path_string);
    // hack because there's no `IntoIterator` impl for `[T; N]`
    let files = files.chain(Some("src/lib.rs".to_owned()).into_iter());

    let (count, fails) = check_files(files);

    // Display results
    println!("Ran {} idempotent tests.", count);
    assert!(fails == 0, "{} idempotent tests failed", fails);
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
        if let Err(msg) = idempotent_check(file_name) {
            print_mismatches(msg);
            fails += 1;
        }
        count += 1;
    }

    (count, fails)
}

fn print_mismatches(result: HashMap<String, Vec<Mismatch>>) {
    let mut t = term::stdout().unwrap();

    for (file_name, diff) in result {
        print_diff(diff, |line_num| format!("\nMismatch at {}:{}:", file_name, line_num));
    }

    assert!(t.reset().unwrap());
}

// Ick, just needed to get a &'static to handle_result.
static HANDLE_RESULT: &'static Fn(HashMap<String, String>) = &handle_result;

pub fn idempotent_check(filename: String) -> Result<(), HashMap<String, Vec<Mismatch>>> {
    let sig_comments = read_significant_comments(&filename);
    let mut config = get_config(sig_comments.get("config").map(|x| &(*x)[..]));
    let args = vec!["rustfmt".to_owned(), filename];

    for (key, val) in sig_comments {
        if key != "target" && key != "config" {
            config.override_value(&key, &val);
        }
    }

    // this thread is not used for concurrency, but rather to workaround the issue that the passed
    // function handle needs to have static lifetime. Instead of using a global RefCell, we use
    // panic to return a result in case of failure. This has the advantage of smoothing the road to
    // multithreaded rustfmt
    thread::catch_panic(move || {
        run(args, WriteMode::Return(HANDLE_RESULT), config);
    }).map_err(|any| *any.downcast().ok().expect("Downcast failed."))
}


// Reads test config file from comments and reads its contents.
fn get_config(config_file: Option<&str>) -> Box<Config> {
    let config_file_name = match config_file {
        None => return Box::new(Default::default()),
        Some(file_name) => {
            let mut full_path = "tests/config/".to_owned();
            full_path.push_str(&file_name);
            full_path
        }
    };

    let mut def_config_file = fs::File::open(config_file_name).ok().expect("Couldn't open config.");
    let mut def_config = String::new();
    def_config_file.read_to_string(&mut def_config).ok().expect("Couldn't read config.");

    Box::new(Config::from_toml(&def_config))
}

// Reads significant comments of the form: // rustfmt-key: value
// into a hash map.
fn read_significant_comments(file_name: &str) -> HashMap<String, String> {
    let file = fs::File::open(file_name).ok().expect(&format!("Couldn't read file {}.", file_name));
    let reader = BufReader::new(file);
    let pattern = r"^\s*//\s*rustfmt-([^:]+):\s*(\S+)";
    let regex = regex::Regex::new(&pattern).ok().expect("Failed creating pattern 1.");

    // Matches lines containing significant comments or whitespace.
    let line_regex = regex::Regex::new(r"(^\s*$)|(^\s*//\s*rustfmt-[^:]+:\s*\S+)").ok()
                         .expect("Failed creating pattern 2.");

    reader.lines()
        .map(|line| line.ok().expect("Failed getting line."))
        .take_while(|line| line_regex.is_match(&line))
        .filter_map(|line| {
            regex.captures_iter(&line)
                .next()
                .map(|capture| {
                    (capture.at(1).expect("Couldn't unwrap capture.").to_owned(),
                     capture.at(2).expect("Couldn't unwrap capture.").to_owned())
                })
        })
        .collect()
}

// Compare output to input.
// TODO: needs a better name, more explanation.
fn handle_result(result: HashMap<String, String>) {
    let mut failures = HashMap::new();

    for (file_name, fmt_text) in result {
        // FIXME: reading significant comments again. Is there a way we can just
        // pass the target to this function?
        let sig_comments = read_significant_comments(&file_name);

        // If file is in tests/source, compare to file with same name in tests/target.
        let target = get_target(&file_name, sig_comments.get("target").map(|x| &(*x)[..]));
        let mut f = fs::File::open(&target).ok().expect("Couldn't open target.");

        let mut text = String::new();
        // TODO: speedup by running through bytes iterator
        f.read_to_string(&mut text).ok().expect("Failed reading target.");
        if fmt_text != text {
            let diff = make_diff(&text, &fmt_text, DIFF_CONTEXT_SIZE);
            failures.insert(file_name, diff);
        }
    }

    if !failures.is_empty() {
        panic!(failures);
    }
}

// Map source file paths to their target paths.
fn get_target(file_name: &str, target: Option<&str>) -> String {
    if file_name.starts_with("tests/source/") {
        let base = target.unwrap_or(file_name.trim_left_matches("tests/source/"));

        format!("tests/target/{}", base)
    } else {
        file_name.to_owned()
    }
}
