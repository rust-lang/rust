// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Tidy checks source code in this repository
//!
//! This program runs all of the various tidy checks for style, cleanliness,
//! etc. This is run by default on `make check` and as part of the auto
//! builders.

#![deny(warnings)]

extern crate tidy;
use tidy::*;

use std::process;
use std::path::PathBuf;
use std::env;

fn main() {
    let path = env::args_os().skip(1).next().expect("need an argument");
    let path = PathBuf::from(path);

    let args: Vec<String> = env::args().skip(1).collect();

    let mut bad = false;
    let quiet = args.iter().any(|s| *s == "--quiet");
    bins::check(&path, &mut bad);
    style::check(&path, &mut bad);
    errors::check(&path, &mut bad);
    cargo::check(&path, &mut bad);
    features::check(&path, &mut bad, quiet);
    pal::check(&path, &mut bad);
    unstable_book::check(&path, &mut bad);
    if !args.iter().any(|s| *s == "--no-vendor") {
        deps::check(&path, &mut bad);
    }

    if bad {
        eprintln!("some tidy checks failed");
        process::exit(1);
    }
}
