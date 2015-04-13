extern crate compiletest;

use std::env;
use std::process::Command;
use std::path::PathBuf;

fn run_mode(mode: &'static str) {

    let mut config = compiletest::default_config();
    let cfg_mode = mode.parse().ok().expect("Invalid mode");
    config.target_rustcflags = Some("-L target/debug/".to_string());

    config.mode = cfg_mode;
    config.src_base = PathBuf::from(format!("tests/{}", mode));

    compiletest::run_tests(&config);
}

#[test]
fn compile_test() {
    run_mode("compile-fail");
    // run_mode("run-pass");
}
