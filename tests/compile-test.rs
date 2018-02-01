#![feature(test)]

extern crate compiletest_rs as compiletest;
extern crate test;

use std::path::{PathBuf, Path};
use std::env::{set_var, var};

fn clippy_driver_path() -> PathBuf {
    if let Some(path) = option_env!("CLIPPY_DRIVER_PATH") {
        PathBuf::from(path)
    } else {
        PathBuf::from(concat!("target/", env!("PROFILE"), "/clippy-driver"))
    }
}

fn host_libs() -> PathBuf {
    if let Some(path) = option_env!("HOST_LIBS") {
        PathBuf::from(path)
    } else {
        Path::new("target").join(env!("PROFILE"))
    }
}

fn rustc_test_suite() -> Option<PathBuf> {
    option_env!("RUSTC_TEST_SUITE").map(PathBuf::from)
}

fn rustc_lib_path() -> PathBuf {
    option_env!("RUSTC_LIB_PATH").unwrap().into()
}

fn config(dir: &'static str, mode: &'static str) -> compiletest::Config {
    let mut config = compiletest::Config::default();

    let cfg_mode = mode.parse().expect("Invalid mode");
    if let Ok(name) = var::<&str>("TESTNAME") {
        let s: String = name.to_owned();
        config.filter = Some(s)
    }

    if rustc_test_suite().is_some() {
        config.run_lib_path = rustc_lib_path();
        config.compile_lib_path = rustc_lib_path();
    }
    config.target_rustcflags = Some(format!("-L {0} -L {0}/deps -Dwarnings", host_libs().display()));

    config.mode = cfg_mode;
    config.build_base = if rustc_test_suite().is_some() {
        // we don't need access to the stderr files on travis
        let mut path = PathBuf::from(env!("OUT_DIR"));
        path.push("test_build_base");
        path
    } else {
        let mut path = std::env::current_dir().unwrap();
        path.push("target/debug/test_build_base");
        path
    };
    config.src_base = PathBuf::from(format!("tests/{}", dir));
    config.rustc_path = clippy_driver_path();
    config
}

fn run_mode(dir: &'static str, mode: &'static str) {
    compiletest::run_tests(&config(dir, mode));
}

fn prepare_env() {
    set_var("CLIPPY_DISABLE_DOCS_LINKS", "true");
    set_var("CLIPPY_TESTS", "true");
    set_var("RUST_BACKTRACE", "0");
}

#[test]
fn compile_test() {
    prepare_env();
    run_mode("run-pass", "run-pass");
    run_mode("ui", "ui");
}
