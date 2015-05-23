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

use std::collections::HashMap;
use std::fs;
use std::io::Read;
use std::thread;
use rustfmt::*;

// For now, the only supported regression tests are idempotent tests - the input and
// output must match exactly.
// FIXME(#28) would be good to check for error messages and fail on them, or at least report.
#[test]
fn idempotent_tests() {
    println!("Idempotent tests:");

    // Get all files in the tests/idem directory
    let files = fs::read_dir("tests/idem").unwrap();
    let files = files.chain(fs::read_dir("tests").unwrap());
    let files = files.chain(fs::read_dir("src/bin").unwrap());
    // turn a DirEntry into a String that represents the relative path to the file
    let files = files.map(|e| e.unwrap().path().to_str().unwrap().to_owned());
    // hack because there's no `IntoIterator` impl for `[T; N]`
    let files = files.chain(Some("src/lib.rs".to_owned()).into_iter());

    // For each file, run rustfmt and collect the output
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

    // Display results
    println!("Ran {} idempotent tests; {} failures.", count, fails);
    assert!(fails == 0, "{} idempotent tests failed", fails);
}

// Compare output to input.
fn print_mismatches(result: HashMap<String, String>) {
    for (file_name, fmt_text) in result {
        println!("{}", fmt_text);
    }
}

// Ick, just needed to get a &'static to handle_result.
static HANDLE_RESULT: &'static Fn(HashMap<String, String>) = &handle_result;

pub fn idempotent_check(filename: String) -> Result<(), HashMap<String, String>> {
    let args = vec!["rustfmt".to_owned(), filename];
    let mut def_config_file = fs::File::open("default.toml").unwrap();
    let mut def_config = String::new();
    def_config_file.read_to_string(&mut def_config).unwrap();
    // this thread is not used for concurrency, but rather to workaround the issue that the passed
    // function handle needs to have static lifetime. Instead of using a global RefCell, we use
    // panic to return a result in case of failure. This has the advantage of smoothing the road to
    // multithreaded rustfmt
    thread::catch_panic(move || {
        run(args, WriteMode::Return(HANDLE_RESULT), &def_config);
    }).map_err(|any|
        *any.downcast().unwrap()
    )
}

// Compare output to input.
fn handle_result(result: HashMap<String, String>) {
    let mut failures = HashMap::new();

    for (file_name, fmt_text) in result {
        let mut f = fs::File::open(&file_name).unwrap();
        let mut text = String::new();
        // TODO: speedup by running through bytes iterator
        f.read_to_string(&mut text).unwrap();
        if fmt_text != text {
            let diff_str = make_diff(&file_name, &fmt_text, &text);
            failures.insert(file_name, diff_str);
        }
    }
    if !failures.is_empty() {
        panic!(failures);
    }
}


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
