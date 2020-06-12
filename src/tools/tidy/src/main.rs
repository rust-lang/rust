//! Tidy checks source code in this repository.
//!
//! This program runs all of the various tidy checks for style, cleanliness,
//! etc. This is run by default on `./x.py test` and as part of the auto
//! builders. The tidy checks can be executed with `./x.py test tidy`.

use tidy::*;

use std::env;
use std::path::PathBuf;
use std::process;

fn main() {
    let path: PathBuf = env::args_os().nth(1).expect("need path to src").into();
    let cargo: PathBuf = env::args_os().nth(2).expect("need path to cargo").into();

    let library_path: PathBuf = path
        .join("..")
        .join("library")
        .canonicalize()
        .expect("unable to canonicalize path to library/");

    let args: Vec<String> = env::args().skip(1).collect();

    let mut bad = false;
    let verbose = args.iter().any(|s| *s == "--verbose");

    // Checks that only make sense for the compiler.
    debug_artifacts::check(&path, &mut bad);
    errors::check(&path, &mut bad);
    ui_tests::check(&path, &mut bad);
    error_codes_check::check(&path, &mut bad);

    // Checks that only make sense for the std libs.
    pal::check(&library_path, &mut bad);
    unit_tests::check(&library_path, &mut bad);

    // Check that need to be done for both the compiler and std libraries.
    bins::check(&path, &mut bad);
    bins::check(&library_path, &mut bad);
    style::check(&path, &mut bad);
    style::check(&library_path, &mut bad);
    cargo::check(&path, &mut bad);
    cargo::check(&library_path, &mut bad);
    edition::check(&path, &mut bad);
    edition::check(&library_path, &mut bad);

    let collected = features::check(&path, &library_path, &mut bad, verbose);
    unstable_book::check(&path, collected, &mut bad);
    deps::check(&path.parent().unwrap(), &cargo, &mut bad);
    extdeps::check(&path.parent().unwrap(), &mut bad);

    if bad {
        eprintln!("some tidy checks failed");
        process::exit(1);
    }
}
