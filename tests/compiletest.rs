extern crate compiletest_rs as compiletest;

use std::path::{PathBuf, Path};
use std::io::Write;

fn compile_fail(sysroot: &str) {
    let flags = format!("--sysroot {} -Dwarnings", sysroot);
    for_all_targets(sysroot, |target| {
        let mut config = compiletest::default_config();
        config.host_rustcflags = Some(flags.clone());
        config.mode = "compile-fail".parse().expect("Invalid mode");
        config.run_lib_path = Path::new(sysroot).join("lib").join("rustlib").join(&target).join("lib");
        config.rustc_path = "target/debug/miri".into();
        config.src_base = PathBuf::from("tests/compile-fail".to_string());
        config.target = target.to_owned();
        config.target_rustcflags = Some(flags.clone());
        compiletest::run_tests(&config);
    });
}

fn run_pass() {
    let mut config = compiletest::default_config();
    config.mode = "run-pass".parse().expect("Invalid mode");
    config.src_base = PathBuf::from("tests/run-pass".to_string());
    config.target_rustcflags = Some("-Dwarnings".to_string());
    config.host_rustcflags = Some("-Dwarnings".to_string());
    compiletest::run_tests(&config);
}

fn for_all_targets<F: FnMut(String)>(sysroot: &str, mut f: F) {
    for target in std::fs::read_dir(format!("{}/lib/rustlib/", sysroot)).unwrap() {
        let target = target.unwrap();
        if !target.metadata().unwrap().is_dir() {
            continue;
        }
        let target = target.file_name().into_string().unwrap();
        if target == "etc" {
            continue;
        }
        let stderr = std::io::stderr();
        writeln!(stderr.lock(), "running tests for target {}", target).unwrap();
        f(target);
    }
}

#[test]
fn compile_test() {
    // Taken from https://github.com/Manishearth/rust-clippy/pull/911.
    let home = option_env!("RUSTUP_HOME").or(option_env!("MULTIRUST_HOME"));
    let toolchain = option_env!("RUSTUP_TOOLCHAIN").or(option_env!("MULTIRUST_TOOLCHAIN"));
    let sysroot = match (home, toolchain) {
        (Some(home), Some(toolchain)) => format!("{}/toolchains/{}", home, toolchain),
        _ => option_env!("RUST_SYSROOT")
            .expect("need to specify RUST_SYSROOT env var or use rustup or multirust")
            .to_owned(),
    };
    compile_fail(&sysroot);
    run_pass();
    for_all_targets(&sysroot, |target| {
        let files = std::fs::read_dir("tests/run-pass").unwrap();
        let files: Box<Iterator<Item=_>> = if let Ok(path) = std::env::var("MIRI_RUSTC_TEST") {
            Box::new(files.chain(std::fs::read_dir(path).unwrap()))
        } else {
            Box::new(files)
        };
        let mut mir_not_found = 0;
        let mut crate_not_found = 0;
        let mut success = 0;
        let mut failed = 0;
        for file in files {
            let file = file.unwrap();
            let path = file.path();

            if !file.metadata().unwrap().is_file() || !path.to_str().unwrap().ends_with(".rs") {
                continue;
            }

            let stderr = std::io::stderr();
            write!(stderr.lock(), "test [miri-pass] {} ... ", path.display()).unwrap();
            let mut cmd = std::process::Command::new("target/debug/miri");
            cmd.arg(path);
            cmd.arg(format!("--target={}", target));
            let libs = Path::new(&sysroot).join("lib");
            let sysroot = libs.join("rustlib").join(&target).join("lib");
            let paths = std::env::join_paths(&[libs, sysroot]).unwrap();
            cmd.env(compiletest::procsrv::dylib_env_var(), paths);

            match cmd.output() {
                Ok(ref output) if output.status.success() => {
                    success += 1;
                    writeln!(stderr.lock(), "ok").unwrap()
                },
                Ok(output) => {
                    let output_err = std::str::from_utf8(&output.stderr).unwrap();
                    if let Some(text) = output_err.splitn(2, "thread 'main' panicked at 'no mir for `").nth(1) {
                        mir_not_found += 1;
                        let end = text.find('`').unwrap();
                        writeln!(stderr.lock(), "NO MIR FOR `{}`", &text[..end]).unwrap();
                    } else if let Some(text) = output_err.splitn(2, "can't find crate for `").nth(1) {
                        crate_not_found += 1;
                        let end = text.find('`').unwrap();
                        writeln!(stderr.lock(), "CAN'T FIND CRATE FOR `{}`", &text[..end]).unwrap();
                    } else {
                        failed += 1;
                        writeln!(stderr.lock(), "FAILED with exit code {:?}", output.status.code()).unwrap();
                        writeln!(stderr.lock(), "stdout: \n {}", std::str::from_utf8(&output.stdout).unwrap()).unwrap();
                        writeln!(stderr.lock(), "stderr: \n {}", output_err).unwrap();
                        panic!("failed to run test");
                    }
                }
                Err(e) => {
                    writeln!(stderr.lock(), "FAILED: {}", e).unwrap();
                    panic!("failed to execute miri");
                },
            }
        }
        let stderr = std::io::stderr();
        writeln!(stderr.lock(), "{} success, {} mir not found, {} crate not found, {} failed", success, mir_not_found, crate_not_found, failed).unwrap();
    });
}
