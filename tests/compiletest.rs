#![feature(slice_concat_ext, custom_test_frameworks)]
// Custom test runner, to avoid libtest being wrapped around compiletest which wraps libtest.
#![test_runner(test_runner)]

use std::slice::SliceConcatExt;
use std::path::PathBuf;
use std::env;

use compiletest_rs as compiletest;
use colored::*;

fn miri_path() -> PathBuf {
    if rustc_test_suite().is_some() {
        PathBuf::from(option_env!("MIRI_PATH").unwrap())
    } else {
        PathBuf::from(concat!("target/", env!("PROFILE"), "/miri"))
    }
}

fn rustc_test_suite() -> Option<PathBuf> {
    option_env!("RUSTC_TEST_SUITE").map(PathBuf::from)
}

fn rustc_lib_path() -> PathBuf {
    option_env!("RUSTC_LIB_PATH").unwrap().into()
}

fn mk_config(mode: &str) -> compiletest::common::ConfigWithTemp {
    let mut config = compiletest::Config::default().tempdir();
    config.mode = mode.parse().expect("Invalid mode");
    config.rustc_path = miri_path();
    if rustc_test_suite().is_some() {
        config.run_lib_path = rustc_lib_path();
        config.compile_lib_path = rustc_lib_path();
    }
    config.filter = env::args().nth(1);
    config.host = get_host();
    config
}

fn compile_fail(path: &str, target: &str, opt: bool) {
    let opt_str = if opt { " with optimizations" } else { "" };
    eprintln!("{}", format!(
        "## Running compile-fail tests in {} against miri for target {}{}",
        path,
        target,
        opt_str
    ).green().bold());

    let mut flags = Vec::new();
    flags.push("-Dwarnings -Dunused".to_owned()); // overwrite the -Aunused in compiletest-rs
    flags.push("--edition 2018".to_owned());
    if opt {
        // Optimizing too aggressivley makes UB detection harder, but test at least
        // the default value.
        // FIXME: Opt level 3 ICEs during stack trace generation.
        flags.push("-Zmir-opt-level=1".to_owned());
    }

    let mut config = mk_config("compile-fail");
    config.src_base = PathBuf::from(path);
    config.target = target.to_owned();
    config.target_rustcflags = Some(flags.join(" "));
    compiletest::run_tests(&config);
}

fn miri_pass(path: &str, target: &str, opt: bool) {
    let opt_str = if opt { " with optimizations" } else { "" };
    eprintln!("{}", format!(
        "## Running run-pass tests in {} against miri for target {}{}",
        path,
        target,
        opt_str
    ).green().bold());

    let mut flags = Vec::new();
    flags.push("-Dwarnings -Dunused".to_owned()); // overwrite the -Aunused in compiletest-rs
    flags.push("--edition 2018".to_owned());
    if opt {
        flags.push("-Zmir-opt-level=3".to_owned());
    }

    let mut config = mk_config("ui");
    config.src_base = PathBuf::from(path);
    config.target = target.to_owned();
    config.target_rustcflags = Some(flags.join(" "));
    compiletest::run_tests(&config);
}

fn get_host() -> String {
    let rustc = rustc_test_suite().unwrap_or(PathBuf::from("rustc"));
    let rustc_version = std::process::Command::new(rustc)
        .arg("-vV")
        .output()
        .expect("rustc not found for -vV")
        .stdout;
    let rustc_version = std::str::from_utf8(&rustc_version).expect("rustc -vV is not utf8");
    let version_meta = rustc_version::version_meta_for(&rustc_version)
        .expect("failed to parse rustc version info");
    version_meta.host
}

fn get_target() -> String {
    std::env::var("MIRI_TEST_TARGET").unwrap_or_else(|_| get_host())
}

fn run_pass_miri(opt: bool) {
    miri_pass("tests/run-pass", &get_target(), opt);
}

fn compile_fail_miri(opt: bool) {
    compile_fail("tests/compile-fail", &get_target(), opt);
}

fn test_runner(_tests: &[&()]) {
    run_pass_miri(false);
    run_pass_miri(true);

    compile_fail_miri(false);
    compile_fail_miri(true);
}
