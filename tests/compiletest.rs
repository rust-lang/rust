#![feature(slice_concat_ext)]

extern crate compiletest_rs as compiletest;
extern crate colored;

use colored::*;

use std::slice::SliceConcatExt;
use std::path::{PathBuf, Path};
use std::io::Write;
use std::env;

macro_rules! eprintln {
    ($($arg:tt)*) => {
        let stderr = std::io::stderr();
        writeln!(stderr.lock(), $($arg)*).unwrap();
    }
}

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

fn have_fullmir() -> bool {
    // We assume we have full MIR when MIRI_SYSROOT is set or when we are in rustc
    std::env::var("MIRI_SYSROOT").is_ok() || rustc_test_suite().is_some()
}

fn compile_fail(sysroot: &Path, path: &str, target: &str, host: &str, need_fullmir: bool) {
    if need_fullmir && !have_fullmir() {
        eprintln!("{}", format!(
            "## Skipping compile-fail tests in {} against miri for target {} due to missing mir",
            path,
            target
        ).yellow().bold());
        return;
    }

    eprintln!("{}", format!(
        "## Running compile-fail tests in {} against miri for target {}",
        path,
        target
    ).green().bold());
    let mut config = compiletest::Config::default().tempdir();
    config.mode = "compile-fail".parse().expect("Invalid mode");
    config.rustc_path = miri_path();
    let mut flags = Vec::new();
    if rustc_test_suite().is_some() {
        config.run_lib_path = rustc_lib_path();
        config.compile_lib_path = rustc_lib_path();
    }
    flags.push(format!("--sysroot {}", sysroot.display()));
    config.src_base = PathBuf::from(path.to_string());
    flags.push("-Zmir-emit-validate=1".to_owned());
    config.target_rustcflags = Some(flags.join(" "));
    config.target = target.to_owned();
    config.host = host.to_owned();
    compiletest::run_tests(&config);
}

fn rustc_pass(sysroot: &Path, path: &str) {
    eprintln!("{}", format!("## Running run-pass tests in {} against rustc", path).green().bold());
    let mut config = compiletest::Config::default().tempdir();
    config.mode = "run-pass".parse().expect("Invalid mode");
    config.src_base = PathBuf::from(path);
    if let Some(rustc_path) = rustc_test_suite() {
        config.rustc_path = rustc_path;
        config.run_lib_path = rustc_lib_path();
        config.compile_lib_path = rustc_lib_path();
        config.target_rustcflags = Some(format!("-Dwarnings --sysroot {}", sysroot.display()));
    } else {
        config.target_rustcflags = Some("-Dwarnings".to_owned());
    }
    config.host_rustcflags = Some("-Dwarnings".to_string());
    compiletest::run_tests(&config);
}

fn miri_pass(sysroot: &Path, path: &str, target: &str, host: &str, need_fullmir: bool, opt: bool) {
    if need_fullmir && !have_fullmir() {
        eprintln!("{}", format!(
            "## Skipping run-pass tests in {} against miri for target {} due to missing mir",
            path,
            target
        ).yellow().bold());
        return;
    }

    let opt_str = if opt { " with optimizations" } else { "" };
    eprintln!("{}", format!(
        "## Running run-pass tests in {} against miri for target {}{}",
        path,
        target,
        opt_str
    ).green().bold());
    let mut config = compiletest::Config::default().tempdir();
    config.mode = "ui".parse().expect("Invalid mode");
    config.src_base = PathBuf::from(path);
    config.target = target.to_owned();
    config.host = host.to_owned();
    config.rustc_path = miri_path();
    if rustc_test_suite().is_some() {
        config.run_lib_path = rustc_lib_path();
        config.compile_lib_path = rustc_lib_path();
    }
    let mut flags = Vec::new();
    flags.push(format!("--sysroot {}", sysroot.display()));
    if have_fullmir() {
        flags.push("-Zmiri-start-fn".to_owned());
    }
    if opt {
        flags.push("-Zmir-opt-level=3".to_owned());
    } else {
        flags.push("-Zmir-opt-level=0".to_owned());
        // For now, only validate without optimizations.  Inlining breaks validation.
        flags.push("-Zmir-emit-validate=1".to_owned());
    }
    // Control miri logging. This is okay despite concurrent test execution as all tests
    // will set this env var to the same value.
    env::set_var("MIRI_LOG", "warn");
    config.target_rustcflags = Some(flags.join(" "));
    compiletest::run_tests(&config);
}

fn is_target_dir<P: Into<PathBuf>>(path: P) -> bool {
    let mut path = path.into();
    path.push("lib");
    path.metadata().map(|m| m.is_dir()).unwrap_or(false)
}

fn for_all_targets<F: FnMut(String)>(sysroot: &Path, mut f: F) {
    let target_dir = sysroot.join("lib").join("rustlib");
    for entry in std::fs::read_dir(target_dir).expect("invalid sysroot") {
        let entry = entry.unwrap();
        if !is_target_dir(entry.path()) {
            continue;
        }
        let target = entry.file_name().into_string().unwrap();
        f(target);
    }
}

fn get_sysroot() -> PathBuf {
    let sysroot = std::env::var("MIRI_SYSROOT").unwrap_or_else(|_| {
        let sysroot = std::process::Command::new("rustc")
            .arg("--print")
            .arg("sysroot")
            .output()
            .expect("rustc not found")
            .stdout;
        String::from_utf8(sysroot).expect("sysroot is not utf8")
    });
    PathBuf::from(sysroot.trim())
}

fn get_host() -> String {
    let rustc = rustc_test_suite().unwrap_or(PathBuf::from("rustc"));
    println!("using rustc at {}", rustc.display());
    let host = std::process::Command::new(rustc)
        .arg("-vV")
        .output()
        .expect("rustc not found for -vV")
        .stdout;
    let host = std::str::from_utf8(&host).expect("sysroot is not utf8");
    let host = host.split("\nhost: ").nth(1).expect(
        "no host: part in rustc -vV",
    );
    let host = host.split('\n').next().expect("no \n after host");
    String::from(host)
}

fn run_pass_miri(opt: bool) {
    let sysroot = get_sysroot();
    let host = get_host();

    for_all_targets(&sysroot, |target| {
        miri_pass(&sysroot, "tests/run-pass", &target, &host, false, opt);
    });
    miri_pass(&sysroot, "tests/run-pass-fullmir", &host, &host, true, opt);
}

fn run_pass_rustc() {
    let sysroot = get_sysroot();
    rustc_pass(&sysroot, "tests/run-pass");
    rustc_pass(&sysroot, "tests/run-pass-fullmir");
}

fn compile_fail_miri() {
    let sysroot = get_sysroot();
    let host = get_host();

    // FIXME: run tests for other targets, too
    compile_fail(&sysroot, "tests/compile-fail", &host, &host, false);
    compile_fail(&sysroot, "tests/compile-fail-fullmir", &host, &host, true);
}

#[test]
fn test() {
    // We put everything into a single test to avoid the parallelism `cargo test`
    // introduces.  We still get parallelism within our tests because `compiletest`
    // uses `libtest` which runs jobs in parallel.

    run_pass_rustc();

    run_pass_miri(false);

    // FIXME: Disabled for now, as the optimizer is pretty broken and crashes...
    // See https://github.com/rust-lang/rust/issues/50411
    //run_pass_miri(true);

    compile_fail_miri();
}
