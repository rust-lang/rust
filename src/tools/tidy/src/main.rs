// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Tidy checks for source code in this repository
//!
//! This program runs all of the various tidy checks for style, cleanliness,
//! etc. This is run by default on `make check` and as part of the auto
//! builders.

use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::{PathBuf, Path};
use std::process;

macro_rules! t {
    ($e:expr, $p:expr) => (match $e {
        Ok(e) => e,
        Err(e) => panic!("{} failed on {} with {}", stringify!($e), ($p).display(), e),
    });

    ($e:expr) => (match $e {
        Ok(e) => e,
        Err(e) => panic!("{} failed with {}", stringify!($e), e),
    })
}

macro_rules! tidy_error {
    ($bad:expr, $fmt:expr, $($arg:tt)*) => ({
        use std::io::Write;
        *$bad = true;
        write!(::std::io::stderr(), "tidy error: ").expect("could not write to stderr");
        writeln!(::std::io::stderr(), $fmt, $($arg)*).expect("could not write to stderr");
    });
}

mod bins;
mod style;
mod errors;
mod features;
mod cargo;
mod pal;
mod deps;
mod unstable_book;

fn main() {
    let path = env::args_os().skip(1).next().expect("need an argument");
    let path = PathBuf::from(path);

    let args: Vec<String> = env::args().skip(1).collect();

    let mut bad = false;
    bins::check(&path, &mut bad);
    style::check(&path, &mut bad);
    errors::check(&path, &mut bad);
    cargo::check(&path, &mut bad);
    features::check(&path, &mut bad);
    pal::check(&path, &mut bad);
    unstable_book::check(&path, &mut bad);
    if !args.iter().any(|s| *s == "--no-vendor") {
        deps::check(&path, &mut bad);
    }

    if bad {
        writeln!(io::stderr(), "some tidy checks failed").expect("could not write to stderr");
        process::exit(1);
    }
}

fn filter_dirs(path: &Path) -> bool {
    let skip = [
        "src/jemalloc",
        "src/llvm",
        "src/libbacktrace",
        "src/compiler-rt",
        "src/rustllvm",
        "src/rust-installer",
        "src/liblibc",
        "src/vendor",
        "src/rt/hoedown",
        "src/tools/cargo",
        "src/tools/rls",
    ];
    skip.iter().any(|p| path.ends_with(p))
}

fn walk_many(paths: &[&Path], skip: &mut FnMut(&Path) -> bool, f: &mut FnMut(&Path)) {
    for path in paths {
        walk(path, skip, f);
    }
}

fn walk(path: &Path, skip: &mut FnMut(&Path) -> bool, f: &mut FnMut(&Path)) {
    for entry in t!(fs::read_dir(path), path) {
        let entry = t!(entry);
        let kind = t!(entry.file_type());
        let path = entry.path();
        if kind.is_dir() {
            if !skip(&path) {
                walk(&path, skip, f);
            }
        } else {
            f(&path);
        }
    }
}
