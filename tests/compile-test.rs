extern crate compiletest_rs as compiletest;

use std::path::PathBuf;
use std::env::var;

fn run_mode(mode: &'static str) {
    let mut config = compiletest::default_config();

    let cfg_mode = mode.parse().ok().expect("Invalid mode");
    config.target_rustcflags = Some("-L target/debug/ -L target/debug/deps".to_owned());
    if let Ok(name) = var::<&str>("TESTNAME") {
        let s : String = name.to_owned();
        config.filter = Some(s)
    }

    config.mode = cfg_mode;
    config.src_base = PathBuf::from(format!("tests/{}", mode));

    compiletest::run_tests(&config);
}

#[test]
#[cfg(not(feature = "test-regex_macros"))]
fn compile_test() {
    run_mode("run-pass");
    run_mode("compile-fail");
}

#[test]
#[cfg(feature = "test-regex_macros")]
fn compile_test() {
    run_mode("run-pass-regex_macros");
    run_mode("compile-fail-regex_macros");
}
