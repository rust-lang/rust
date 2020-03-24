#![feature(custom_test_frameworks)]
// Custom test runner, to avoid libtest being wrapped around compiletest which wraps libtest.
#![test_runner(test_runner)]

use std::env;
use std::path::PathBuf;

use colored::*;
use compiletest_rs as compiletest;

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

fn run_tests(mode: &str, path: &str, target: &str) {
    let in_rustc_test_suite = rustc_test_suite().is_some();
    // Add some flags we always want.
    let mut flags = Vec::new();
    flags.push("--edition 2018".to_owned());
    if in_rustc_test_suite {
        // Less aggressive warnings to make the rustc toolstate management less painful.
        // (We often get warnings when e.g. a feature gets stabilized or some lint gets added/improved.)
        flags.push("-Astable-features".to_owned());
    } else {
        flags.push("-Dwarnings -Dunused".to_owned()); // overwrite the -Aunused in compiletest-rs
    }
    if let Ok(sysroot) = std::env::var("MIRI_SYSROOT") {
        flags.push(format!("--sysroot {}", sysroot));
    }
    if let Ok(extra_flags) = std::env::var("MIRI_TEST_FLAGS") {
        flags.push(extra_flags);
    }

    let flags = flags.join(" ");
    eprintln!("   Compiler flags: {}", flags);

    // The rest of the configuration.
    let mut config = compiletest::Config::default().tempdir();
    config.mode = mode.parse().expect("Invalid mode");
    config.rustc_path = miri_path();
    if in_rustc_test_suite {
        config.run_lib_path = rustc_lib_path();
        config.compile_lib_path = rustc_lib_path();
    }
    config.filter = env::args().nth(1);
    config.host = get_host();
    config.src_base = PathBuf::from(path);
    config.target = target.to_owned();
    config.target_rustcflags = Some(flags);
    compiletest::run_tests(&config);
}

fn compile_fail(path: &str, target: &str) {
    eprintln!(
        "{}",
        format!(
            "## Running compile-fail tests in {} against miri for target {}",
            path, target
        )
        .green()
        .bold()
    );

    run_tests("compile-fail", path, target);
}

fn miri_pass(path: &str, target: &str) {
    eprintln!(
        "{}",
        format!(
            "## Running run-pass tests in {} against miri for target {}",
            path, target
        )
        .green()
        .bold()
    );

    run_tests("ui", path, target);
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

fn test_runner(_tests: &[&()]) {
    // Add a test env var to do environment communication tests.
    std::env::set_var("MIRI_ENV_VAR_TEST", "0");
    // Let the tests know where to store temp files (they might run for a different target, which can make this hard to find).
    std::env::set_var("MIRI_TEMP", std::env::temp_dir());

    let target = get_target();
    miri_pass("tests/run-pass", &target);
    compile_fail("tests/compile-fail", &target);
}
