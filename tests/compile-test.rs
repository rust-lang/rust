extern crate compiletest_rs as compiletest;

use std::path::PathBuf;
use std::env::{set_var, var};

fn run_mode(dir: &'static str, mode: &'static str) {
    let mut config = compiletest::default_config();

    let cfg_mode = mode.parse().ok().expect("Invalid mode");
    config.target_rustcflags = Some("-L target/debug/ -L target/debug/deps".to_owned());
    if let Ok(name) = var::<&str>("TESTNAME") {
        let s: String = name.to_owned();
        config.filter = Some(s)
    }

    config.mode = cfg_mode;
    config.src_base = PathBuf::from(format!("tests/{}", dir));

    compiletest::run_tests(&config);
}

fn prepare_env() {
    set_var("CLIPPY_DISABLE_WIKI_LINKS", "true");
}

#[test]
fn compile_test() {
    prepare_env();
    run_mode("run-pass", "run-pass");
    run_mode("compile-fail", "compile-fail");
}
