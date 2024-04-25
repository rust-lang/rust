//! This checks the output of `--print=native-static-libs`
//!
//! Specifically, this test makes sure that one and only one
//! note is emitted with the text "native-static-libs:" as prefix
//! that the note contains the link args given in the source code
//! and cli of the current crate and downstream crates.
//!
//! It also checks that there aren't any duplicated consecutive
//! args, as they are useless and suboptimal for debugability.
//! See https://github.com/rust-lang/rust/issues/113209.

//@ ignore-cross-compile
//@ ignore-wasm

extern crate run_make_support;

use std::io::BufRead;

use run_make_support::rustc;

fn main() {
    // build supporting crate
    rustc()
        .input("bar.rs")
        .crate_type("rlib")
        .arg("-lbar_cli")
        .run();

    // build main crate as staticlib
    let output = rustc()
        .input("foo.rs")
        .crate_type("staticlib")
        .arg("-lfoo_cli")
        .arg("-lfoo_cli") // 2nd time
        .print("native-static-libs")
        .run();

    let mut found_note = false;
    for l in output.stderr.lines() {
        let l = l.expect("utf-8 string");

        let Some(args) = l.strip_prefix("note: native-static-libs:") else { continue; };
        assert!(!found_note);
        found_note = true;

        let args: Vec<&str> = args.trim().split_ascii_whitespace().collect();

        assert!(args.contains(&"-lglib-2.0")); // in bar.rs
        assert!(args.contains(&"-lsystemd")); // in foo.rs
        assert!(args.contains(&"-lbar_cli"));
        assert!(args.contains(&"-lfoo_cli"));

        // make sure that no args are consecutively present
        let dedup_args: Vec<&str> = {
            let mut args = args.clone();
            args.dedup();
            args
        };
        assert_eq!(args, dedup_args);
    }

    assert!(found_note);
}
