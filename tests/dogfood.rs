#![feature(test, plugin)]
#![plugin(clippy)]
#![deny(clippy, clippy_pedantic)]

extern crate compiletest_rs as compiletest;
extern crate test;

use std::env::{var, set_var};
use std::path::PathBuf;
use test::TestPaths;

#[test]
fn dogfood() {
    // don't run dogfood on travis, cargo-clippy already runs clippy on itself
    if let Ok(travis) = var("TRAVIS") {
        if travis == "true" {
            return;
        }
    }

    let mut config = compiletest::default_config();

    let cfg_mode = "run-fail".parse().expect("Invalid mode");
    let mut s = String::new();
    s.push_str(" -L clippy_tests/target/debug/");
    s.push_str(" -L clippy_tests/target/debug/deps");
    s.push_str(" -Zextra-plugins=clippy -Ltarget_recur/debug -Dwarnings -Dclippy_pedantic -Dclippy -Dclippy_internal");
    config.target_rustcflags = Some(s);
    if let Ok(name) = var("TESTNAME") {
        config.filter = Some(name.to_owned())
    }

    config.mode = cfg_mode;
    config.verbose = true;

    let files = ["src/main.rs", "src/lib.rs", "clippy_lints/src/lib.rs"];

    for file in &files {
        let paths = TestPaths {
            base: PathBuf::new(),
            file: PathBuf::from(file),
            relative_dir: PathBuf::new(),
        };

        set_var("CLIPPY_DOGFOOD", "tastes like chicken");

        compiletest::runtest::run(config.clone(), &paths);
    }
}
