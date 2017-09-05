extern crate compiletest_rs as compiletest;

use std::path::PathBuf;
use std::env::{set_var, var};

fn run_mode(dir: &'static str, mode: &'static str) {
    let mut config = compiletest::Config::default();

    let cfg_mode = mode.parse().expect("Invalid mode");
    config.target_rustcflags = Some("-L target/debug/ -L target/debug/deps -Dwarnings".to_owned());
    if let Ok(name) = var::<&str>("TESTNAME") {
        let s: String = name.to_owned();
        config.filter = Some(s)
    }

    config.mode = cfg_mode;
    config.build_base = PathBuf::from("target/debug/test_build_base");
    config.src_base = PathBuf::from(format!("tests/{}", dir));

    compiletest::run_tests(&config);
}

fn prepare_env() {
    set_var("CLIPPY_DISABLE_DOCS_LINKS", "true");
}

#[test]
fn compile_test() {
    prepare_env();
    run_mode("run-pass", "run-pass");
    run_mode("ui", "ui");
    #[cfg(target_os = "windows")]
    run_mode("ui-windows", "ui");
    #[cfg(not(target_os = "windows"))]
    run_mode("ui-posix", "ui");
}
