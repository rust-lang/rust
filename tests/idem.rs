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

use std::collections::HashMap;
use std::fs;
use std::io::Read;
use std::sync::atomic;
use rustfmt::*;

// For now, the only supported regression tests are idempotent tests - the input and
// output must match exactly.
// FIXME(#28) would be good to check for error messages and fail on them, or at least report.
#[test]
fn idempotent_tests() {
    println!("Idempotent tests:");
    FAILURES.store(0, atomic::Ordering::Relaxed);

    // Get all files in the tests/idem directory
    let files = fs::read_dir("tests/idem").unwrap();
    // For each file, run rustfmt and collect the output
    let mut count = 0;
    for entry in files {
        let path = entry.unwrap().path();
        let file_name = path.to_str().unwrap();
        println!("Testing '{}'...", file_name);
        run(vec!["rustfmt".to_owned(), file_name.to_owned()], WriteMode::Return(HANDLE_RESULT));
        count += 1;
    }
    // And also dogfood ourselves!
    println!("Testing 'src/lib.rs'...");
    run(vec!["rustfmt".to_string(), "src/lib.rs".to_string()],
        WriteMode::Return(HANDLE_RESULT));
    count += 1;

    // Display results
    let fails = FAILURES.load(atomic::Ordering::Relaxed);
    println!("Ran {} idempotent tests; {} failures.", count, fails);
    assert!(fails == 0, "{} idempotent tests failed", fails);
}

// 'global' used by sys_tests and handle_result.
static FAILURES: atomic::AtomicUsize = atomic::ATOMIC_USIZE_INIT;
// Ick, just needed to get a &'static to handle_result.
static HANDLE_RESULT: &'static Fn(HashMap<String, String>) = &handle_result;

// Compare output to input.
fn handle_result(result: HashMap<String, String>) {
    let mut fails = 0;

    for file_name in result.keys() {
        let mut f = fs::File::open(file_name).unwrap();
        let mut text = String::new();
        f.read_to_string(&mut text).unwrap();
        if result[file_name] != text {
            fails += 1;
            println!("Mismatch in {}.", file_name);
            println!("{}", result[file_name]);
        }
    }

    if fails > 0 {
        FAILURES.fetch_add(1, atomic::Ordering::Relaxed);
    }
}