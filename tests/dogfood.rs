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
    let mut config = compiletest::default_config();

    let cfg_mode = "run-pass".parse().expect("Invalid mode");
    let mut s = String::new();
    s.push_str(" -L target/debug/");
    s.push_str(" -L target/debug/deps");
    s.push_str(" -Zextra-plugins=clippy -Ltarget_recur/debug -Dclippy_pedantic -Dclippy");
    config.target_rustcflags = Some(s);
    if let Ok(name) = var("TESTNAME") {
        config.filter = Some(name.to_owned())
    }

    config.mode = cfg_mode;

    let paths = TestPaths {
        base: PathBuf::new(),
        file: PathBuf::from("src/lib.rs"),
        relative_dir: PathBuf::new(),
    };

    set_var("CLIPPY_DOGFOOD", "tastes like chicken");

    compiletest::runtest::run(config, &paths);
}
