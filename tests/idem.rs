// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(std_misc)]

extern crate rustfmt;

use std::collections::HashMap;
use std::fs;
use std::io::Read;
use rustfmt::*;

// For now, the only supported regression tests are idempotent tests - the input and
// output must match exactly.
// FIXME(#28) would be good to check for error messages and fail on them, or at least report.
#[test]
fn idempotent_tests() {
    println!("Idempotent tests:");

    // Get all files in the tests/idem directory
    let files = fs::read_dir("tests/idem").unwrap();
    let files2 = fs::read_dir("tests").unwrap();
    let files3 = fs::read_dir("src/bin").unwrap();
    // For each file, run rustfmt and collect the output

    let mut count = 0;
    let mut fails = 0;
    for entry in files.chain(files2).chain(files3) {
        let path = entry.unwrap().path();
        let file_name = path.to_str().unwrap();
        if !file_name.ends_with(".rs") {
            continue;
        }
        println!("Testing '{}'...", file_name);
        match idempotent_check(vec!["rustfmt".to_owned(), file_name.to_owned()]) {
            Ok(()) => {},
            Err(m) => {
                print_mismatches(m);
                fails += 1;
            },
        }
        count += 1;
    }
    // And also dogfood rustfmt!
    println!("Testing 'src/lib.rs'...");
    match idempotent_check(vec!["rustfmt".to_owned(), "src/lib.rs".to_owned()]) {
        Ok(()) => {},
        Err(m) => {
            print_mismatches(m);
            fails += 1;
        },
    }
    count += 1;

    // Display results
    println!("Ran {} idempotent tests; {} failures.", count, fails);
    assert!(fails == 0, "{} idempotent tests failed", fails);
}

// Compare output to input.
fn print_mismatches(result: HashMap<String, String>) {
    for (file_name, fmt_text) in result {
        println!("Mismatch in {}.", file_name);
        println!("{}", fmt_text);
    }
}

// Ick, just needed to get a &'static to handle_result.
static HANDLE_RESULT: &'static Fn(HashMap<String, String>) = &handle_result;

pub fn idempotent_check(args: Vec<String>) -> Result<(), HashMap<String, String>> {
    use std::thread;
    use std::fs;
    use std::io::Read;
    thread::spawn(move || {
        run(args, WriteMode::Return(HANDLE_RESULT));
    }).join().map_err(|mut any|
        any.downcast_mut::<HashMap<String, String>>()
           .unwrap() // i know it is a hashmap
           .drain() // i only get a reference :(
           .collect() // so i need to turn it into an iter and then back
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
            failures.insert(file_name, fmt_text);
        }
    }
    if !failures.is_empty() {
        panic!(failures);
    }
}