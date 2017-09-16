#![feature(slice_concat_ext)]

extern crate compiletest_rs as compiletest;

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

fn compile_fail(sysroot: &Path, path: &str, target: &str, host: &str, fullmir: bool) {
    eprintln!(
        "## Running compile-fail tests in {} against miri for target {}",
        path,
        target
    );
    let mut config = compiletest::Config::default().tempdir();
    config.mode = "compile-fail".parse().expect("Invalid mode");
    config.rustc_path = miri_path();
    let mut flags = Vec::new();
    if rustc_test_suite().is_some() {
        config.run_lib_path = rustc_lib_path();
        config.compile_lib_path = rustc_lib_path();
    }
    // if we are building as part of the rustc test suite, we already have fullmir for everything
    if fullmir && rustc_test_suite().is_none() {
        if host != target {
            // skip fullmir on nonhost
            return;
        }
        let sysroot = std::env::home_dir().unwrap()
            .join(".xargo")
            .join("HOST");
        config.target_rustcflags = Some(format!("--sysroot {}", sysroot.to_str().unwrap()));
        config.src_base = PathBuf::from(path.to_string());
    } else {
        config.target_rustcflags = Some(format!("--sysroot {}", sysroot.to_str().unwrap()));
        config.src_base = PathBuf::from(path.to_string());
    }
    flags.push("-Zmir-emit-validate=1".to_owned());
    config.target_rustcflags = Some(flags.join(" "));
    config.target = target.to_owned();
    compiletest::run_tests(&config);
}

fn run_pass(path: &str) {
    eprintln!("## Running run-pass tests in {} against rustc", path);
    let mut config = compiletest::Config::default().tempdir();
    config.mode = "run-pass".parse().expect("Invalid mode");
    config.src_base = PathBuf::from(path);
    if let Some(rustc_path) = rustc_test_suite() {
        config.rustc_path = rustc_path;
        config.run_lib_path = rustc_lib_path();
        config.compile_lib_path = rustc_lib_path();
        config.target_rustcflags = Some(format!("-Dwarnings --sysroot {}", get_sysroot().display()));
    } else {
        config.target_rustcflags = Some("-Dwarnings".to_owned());
    }
    config.host_rustcflags = Some("-Dwarnings".to_string());
    compiletest::run_tests(&config);
}

fn miri_pass(path: &str, target: &str, host: &str, fullmir: bool, opt: bool) {
    let opt_str = if opt { " with optimizations" } else { "" };
    eprintln!(
        "## Running run-pass tests in {} against miri for target {}{}",
        path,
        target,
        opt_str
    );
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
    // Control miri logging. This is okay despite concurrent test execution as all tests
    // will set this env var to the same value.
    env::set_var("MIRI_LOG", "warn");
    // if we are building as part of the rustc test suite, we already have fullmir for everything
    if fullmir && rustc_test_suite().is_none() {
        if host != target {
            // skip fullmir on nonhost
            return;
        }
        let sysroot = std::env::home_dir().unwrap()
            .join(".xargo")
            .join("HOST");

        flags.push(format!("--sysroot {}", sysroot.to_str().unwrap()));
    }
    if opt {
        flags.push("-Zmir-opt-level=3".to_owned());
    } else {
        flags.push("-Zmir-opt-level=0".to_owned());
        // For now, only validate without optimizations.  Inlining breaks validation.
        flags.push("-Zmir-emit-validate=1".to_owned());
    }
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
        miri_pass("tests/run-pass", &target, &host, false, opt);
    });
    miri_pass("tests/run-pass-fullmir", &host, &host, true, opt);
}

#[test]
fn run_pass_miri_noopt() {
    run_pass_miri(false);
}

#[test]
#[ignore] // FIXME: Disabled for now, as the optimizer is pretty broken and crashes...
fn run_pass_miri_opt() {
    run_pass_miri(true);
}

#[test]
fn run_pass_rustc() {
    run_pass("tests/run-pass");
    run_pass("tests/run-pass-fullmir");
}

#[test]
fn compile_fail_miri() {
    let sysroot = get_sysroot();
    let host = get_host();

    for_all_targets(&sysroot, |target| {
        compile_fail(&sysroot, "tests/compile-fail", &target, &host, false);
    });
    compile_fail(&sysroot, "tests/compile-fail-fullmir", &host, &host, true);
}
