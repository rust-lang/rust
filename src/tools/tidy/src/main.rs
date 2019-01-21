//! Tidy checks source code in this repository.
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
    let path: PathBuf = env::args_os().nth(1).expect("need path to src").into();
    let cargo: PathBuf = env::args_os().nth(2).expect("need path to cargo").into();

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
    libcoretest::check(&path, &mut bad);
    if !args.iter().any(|s| *s == "--no-vendor") {
        deps::check(&path, &mut bad);
    }
    deps::check_whitelist(&path, &cargo, &mut bad);
    extdeps::check(&path, &mut bad);
    ui_tests::check(&path, &mut bad);

    if bad {
        eprintln!("some tidy checks failed");
        process::exit(1);
    }
}
