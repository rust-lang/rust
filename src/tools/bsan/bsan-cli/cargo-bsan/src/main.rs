#![feature(rustc_private)]
#![warn(clippy::pedantic)]

mod arg;
mod phases;
mod setup;
mod util;

use crate::util::show_error;
use phases::*;

use log::debug;
use std::env;

const CARGO_BSAN_HELP: &str = r"Runs binary crates and tests with BorrowSanitizer enabled.

Usage:
    cargo bsan [subcommand] [<cargo options>...] [--] [<program/test suite options>...]

Subcommands:
    run, r                   Run binaries
    test, t                  Run tests
    nextest                  Run tests with nextest (requires cargo-nextest installed)
    setup                    Only perform automatic setup, but without asking questions (for getting a proper libstd)
    clean                    Clean the BorrowSanitizer cache & target directory

The cargo options are exactly the same as for `cargo run` and `cargo test`, respectively.
Furthermore, the following extra flags and environment variables are recognized for `run` and `test`:

    BSANFLAGS                Extra flags to pass to the driver. Use this to pass `-Zbsan-...` flags.

Examples:
    cargo bsan run
    cargo bsan test -- test-suite-filter

    cargo bsan setup --print-sysroot
        This will print the path to the generated sysroot (and nothing else) on stdout.
        stderr will still contain progress information about how the build is doing.
";

fn show_help() {
    println!("{CARGO_BSAN_HELP}");
}

fn show_version() {
    print!("bsan {}", env!("CARGO_PKG_VERSION"));
    let version = format!("{} {}", env!("GIT_HASH"), env!("COMMIT_DATE"));
    if version.len() > 1 {
        // If there is actually something here, print it.
        print!(" ({version})");
    }
    println!();
}

fn main() {
    env_logger::init();
    let mut args = env::args();
    debug!("args: {:?}", args);

    // Skip binary name.
    args.next().unwrap();

    let Some(first) = args.next() else {
        show_error!(
            "`cargo-bsan` called without first argument; please only invoke this binary through `cargo bsan`"
        )
    };

    match first.as_str() {
        "bsan" => phase_cargo_bsan(args),
        _ => show_help(),
    }
}
