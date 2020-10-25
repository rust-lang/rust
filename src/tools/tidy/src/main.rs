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
    let root_path: PathBuf = env::args_os().nth(1).expect("need path to root of repo").into();
    let cargo: PathBuf = env::args_os().nth(2).expect("need path to cargo").into();
    let output_directory: PathBuf =
        env::args_os().nth(3).expect("need path to output directory").into();

    let src_path = root_path.join("src");
    let library_path = root_path.join("library");
    let compiler_path = root_path.join("compiler");

    let args: Vec<String> = env::args().skip(1).collect();

    let mut bad = false;
    let verbose = args.iter().any(|s| *s == "--verbose");

    // Checks over tests.
    debug_artifacts::check(&src_path, &mut bad);
    ui_tests::check(&src_path, &mut bad);

    // Checks that only make sense for the compiler.
    errors::check(&compiler_path, &mut bad);
    error_codes_check::check(&src_path, &mut bad);

    // Checks that only make sense for the std libs.
    pal::check(&library_path, &mut bad);
    unit_tests::check(&library_path, &mut bad);

    // Checks that need to be done for both the compiler and std libraries.
    bins::check(&src_path, &output_directory, &mut bad);
    bins::check(&compiler_path, &output_directory, &mut bad);
    bins::check(&library_path, &output_directory, &mut bad);

    style::check(&src_path, &mut bad);
    style::check(&compiler_path, &mut bad);
    style::check(&library_path, &mut bad);

    cargo::check(&src_path, &mut bad);
    cargo::check(&compiler_path, &mut bad);
    cargo::check(&library_path, &mut bad);

    edition::check(&src_path, &mut bad);
    edition::check(&compiler_path, &mut bad);
    edition::check(&library_path, &mut bad);

    let collected = features::check(&src_path, &compiler_path, &library_path, &mut bad, verbose);
    unstable_book::check(&src_path, collected, &mut bad);

    // Checks that are done on the cargo workspace.
    deps::check(&root_path, &cargo, &mut bad);
    extdeps::check(&root_path, &mut bad);

    if bad {
        eprintln!("some tidy checks failed");
        process::exit(1);
    }
}
