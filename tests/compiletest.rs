use std::env;
use std::path::PathBuf;

use colored::*;
use compiletest_rs as compiletest;

fn miri_path() -> PathBuf {
    PathBuf::from(option_env!("MIRI").unwrap_or(env!("CARGO_BIN_EXE_miri")))
}

fn run_tests(mode: &str, path: &str, target: &str) {
    let in_rustc_test_suite = option_env!("RUSTC_STAGE").is_some();
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
    if let Ok(sysroot) = env::var("MIRI_SYSROOT") {
        flags.push(format!("--sysroot {}", sysroot));
    }
    if let Ok(extra_flags) = env::var("MIRIFLAGS") {
        flags.push(extra_flags);
    }

    let flags = flags.join(" ");
    eprintln!("   Compiler flags: {}", flags);

    // The rest of the configuration.
    let mut config = compiletest::Config::default().tempdir();
    config.mode = mode.parse().expect("Invalid mode");
    config.rustc_path = miri_path();
    if let Some(lib_path) = option_env!("RUSTC_LIB_PATH") {
        config.run_lib_path = PathBuf::from(lib_path);
        config.compile_lib_path = PathBuf::from(lib_path);
    }
    config.filters = env::args().nth(1).into_iter().collect();
    config.host = get_host();
    config.src_base = PathBuf::from(path);
    config.target = target.to_owned();
    config.target_rustcflags = Some(flags);
    compiletest::run_tests(&config);
}

fn compile_fail(path: &str, target: &str) {
    eprintln!(
        "{}",
        format!("## Running compile-fail tests in {} against miri for target {}", path, target)
            .green()
            .bold()
    );

    run_tests("compile-fail", path, target);
}

fn miri_pass(path: &str, target: &str) {
    eprintln!(
        "{}",
        format!("## Running run-pass tests in {} against miri for target {}", path, target)
            .green()
            .bold()
    );

    run_tests("ui", path, target);
}

fn get_host() -> String {
    let version_meta =
        rustc_version::VersionMeta::for_command(std::process::Command::new(miri_path()))
            .expect("failed to parse rustc version info");
    version_meta.host
}

fn get_target() -> String {
    env::var("MIRI_TEST_TARGET").unwrap_or_else(|_| get_host())
}

fn main() {
    // Add a test env var to do environment communication tests.
    env::set_var("MIRI_ENV_VAR_TEST", "0");
    // Let the tests know where to store temp files (they might run for a different target, which can make this hard to find).
    env::set_var("MIRI_TEMP", env::temp_dir());
    // Panic tests expect backtraces to be printed.
    env::set_var("RUST_BACKTRACE", "1");

    let target = get_target();
    miri_pass("tests/run-pass", &target);
    compile_fail("tests/compile-fail", &target);
}
