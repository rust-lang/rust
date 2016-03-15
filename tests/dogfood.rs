#![feature(test)]

extern crate compiletest_rs as compiletest;
extern crate test;

use std::env::var;
use std::path::PathBuf;
use test::TestPaths;

#[test]
fn dogfood() {
    let mut config = compiletest::default_config();

    let cfg_mode = "run-pass".parse().ok().expect("Invalid mode");
    let mut s = String::new();
    s.push_str(" -L target/debug/");
    s.push_str(" -L target/debug/deps");
    s.push_str(" -Zextra-plugins=clippy -Ltarget_recur/debug -Dclippy_pedantic -Dclippy");
    config.target_rustcflags = Some(s);
    if let Ok(name) = var::<&str>("TESTNAME") {
        let s : String = name.to_owned();
        config.filter = Some(s)
    }

    config.mode = cfg_mode;

    let paths = TestPaths {
        base: PathBuf::new(),
        file: PathBuf::from("src/lib.rs"),
        relative_dir: PathBuf::new(),
    };
    compiletest::runtest::run(config, &paths);
}
