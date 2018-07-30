// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ! Check for external package sources. Allow only vendorable packages.

use std::fs::File;
use std::io::Read;
use std::path::Path;

/// List of whitelisted sources for packages
static WHITELISTED_SOURCES: &'static [&'static str] = &[
    "\"registry+https://github.com/rust-lang/crates.io-index\"",
];

/// check for external package sources
pub fn check(path: &Path, bad: &mut bool) {
    // Cargo.lock of rust: src/Cargo.lock
    let path = path.join("Cargo.lock");

    // open and read the whole file
    let mut cargo_lock = String::new();
    t!(t!(File::open(path)).read_to_string(&mut cargo_lock));

    // process each line
    let mut lines = cargo_lock.lines();
    while let Some(line) = lines.next() {

        // consider only source entries
        if ! line.starts_with("source = ") {
            continue;
        }

        // extract source value
        let parts: Vec<&str> = line.splitn(2, "=").collect();
        let source = parts[1].trim();

        // ensure source is whitelisted
        if !WHITELISTED_SOURCES.contains(&&*source) {
            println!("invalid source: {}", source);
            *bad = true;
        }
    }
}
