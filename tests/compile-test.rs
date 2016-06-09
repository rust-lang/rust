extern crate compiletest_rs as compiletest;

use std::path::PathBuf;
use std::env::{set_var, var, temp_dir};

fn run_mode(dir: &'static str, mode: &'static str) {
    let mut config = compiletest::default_config();

    let cfg_mode = mode.parse().ok().expect("Invalid mode");
    config.target_rustcflags = Some("-L target/debug/ -L target/debug/deps".to_owned());
    if let Ok(name) = var::<&str>("TESTNAME") {
        let s: String = name.to_owned();
        config.filter = Some(s)
    }

    config.mode = cfg_mode;
    if cfg!(windows) {
        // work around https://github.com/laumann/compiletest-rs/issues/35 on msvc windows
        config.build_base = temp_dir();
    }
    config.src_base = PathBuf::from(format!("tests/{}", dir));

    compiletest::run_tests(&config);
}

fn prepare_env() {
    set_var("CLIPPY_DISABLE_WIKI_LINKS", "true");
}

#[test]
#[cfg(not(feature = "test-regex_macros"))]
fn compile_test() {
    prepare_env();
    run_mode("run-pass", "run-pass");
    run_mode("compile-fail", "compile-fail");
}

#[test]
#[cfg(feature = "test-regex_macros")]
fn compile_test() {
    prepare_env();
    run_mode("run-pass-regex_macros", "run-pass");
    run_mode("compile-fail-regex_macros", "compile-fail");
}
