// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Tidy check to ensure that `[dependencies]` and `extern crate` are in sync.
//!
//! This tidy check ensures that all crates listed in the `[dependencies]`
//! section of a `Cargo.toml` are present in the corresponding `lib.rs` as
//! `extern crate` declarations. This should help us keep the DAG correctly
//! structured through various refactorings to prune out unnecessary edges.

use std::io::prelude::*;
use std::fs::File;
use std::path::Path;

pub fn check(path: &Path, bad: &mut bool) {
    if path.ends_with("vendor") {
        return
    }
    for entry in t!(path.read_dir(), path).map(|e| t!(e)) {
        // Look for `Cargo.toml` with a sibling `src/lib.rs` or `lib.rs`
        if entry.file_name().to_str() == Some("Cargo.toml") {
            if path.join("src/lib.rs").is_file() {
                verify(&entry.path(), &path.join("src/lib.rs"), bad)
            }
            if path.join("lib.rs").is_file() {
                verify(&entry.path(), &path.join("lib.rs"), bad)
            }
        } else if t!(entry.file_type()).is_dir() {
            check(&entry.path(), bad);
        }
    }
}

// Verify that the dependencies in Cargo.toml at `tomlfile` are sync'd with the
// `extern crate` annotations in the lib.rs at `libfile`.
fn verify(tomlfile: &Path, libfile: &Path, bad: &mut bool) {
    let mut toml = String::new();
    let mut librs = String::new();
    t!(t!(File::open(tomlfile)).read_to_string(&mut toml));
    t!(t!(File::open(libfile)).read_to_string(&mut librs));

    if toml.contains("name = \"bootstrap\"") {
        return
    }

    // "Poor man's TOML parser", just assume we use one syntax for now
    //
    // We just look for:
    //
    //      [dependencies]
    //      name = ...
    //      name2 = ...
    //      name3 = ...
    //
    // If we encounter a line starting with `[` then we assume it's the end of
    // the dependency section and bail out.
    let deps = match toml.find("[dependencies]") {
        Some(i) => &toml[i+1..],
        None => return,
    };
    let mut lines = deps.lines().peekable();
    while let Some(line) = lines.next() {
        if line.starts_with("[") {
            break
        }

        let mut parts = line.splitn(2, '=');
        let krate = parts.next().unwrap().trim();
        if parts.next().is_none() {
            continue
        }

        // Don't worry about depending on core/std but not saying `extern crate
        // core/std`, that's intentional.
        if krate == "core" || krate == "std" {
            continue
        }

        // This is intentional, this dependency just makes the crate available
        // for others later on. Cover cases
        let whitelisted = krate == "alloc_jemalloc";
        let whitelisted = whitelisted || krate.starts_with("panic");
        if toml.contains("name = \"std\"") && whitelisted {
            continue
        }

        // We want the compiler to depend on the proc_macro_plugin crate so
        // that it is built and included in the end, but we don't want to
        // actually use it in the compiler.
        if toml.contains("name = \"rustc_driver\"") &&
           krate == "proc_macro_plugin" {
            continue
        }

        if !librs.contains(&format!("extern crate {}", krate)) {
            println!("{} doesn't have `extern crate {}`, but Cargo.toml \
                      depends on it", libfile.display(), krate);
            *bad = true;
        }
    }
}
