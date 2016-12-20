extern crate compiletest_rs as compiletest;

use std::path::{PathBuf, Path};
use std::io::Write;

fn compile_fail(sysroot: &Path) {
    let flags = format!("--sysroot {} -Dwarnings", sysroot.to_str().expect("non utf8 path"));
    for_all_targets(&sysroot, |target| {
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

fn miri_pass(path: &str, target: &str, host: &str) {
    let mut config = compiletest::default_config();
    config.mode = "mir-opt".parse().expect("Invalid mode");
    config.src_base = PathBuf::from(path);
    config.target = target.to_owned();
    config.rustc_path = PathBuf::from("target/debug/miri");
    // don't actually execute the final binary, it might be for other targets and we only care
    // about running miri, not the binary.
    config.runtool = Some("echo \"\" || ".to_owned());
    if target == host {
        std::env::set_var("MIRI_HOST_TARGET", "yes");
    }
    compiletest::run_tests(&config);
    std::env::set_var("MIRI_HOST_TARGET", "");
}

fn is_target_dir<P: Into<PathBuf>>(path: P) -> bool {
    let mut path = path.into();
    path.push("lib");
    path.metadata().map(|m| m.is_dir()).unwrap_or(false)
}

fn for_all_targets<F: FnMut(String)>(sysroot: &Path, mut f: F) {
    let target_dir = sysroot.join("lib").join("rustlib");
    println!("target dir: {}", target_dir.to_str().unwrap());
    for entry in std::fs::read_dir(target_dir).expect("invalid sysroot") {
        let entry = entry.unwrap();
        if !is_target_dir(entry.path()) { continue; }
        let target = entry.file_name().into_string().unwrap();
        let stderr = std::io::stderr();
        writeln!(stderr.lock(), "running tests for target {}", target).unwrap();
        f(target);
    }
}

#[test]
fn compile_test() {
    let sysroot = std::process::Command::new("rustc")
        .arg("--print")
        .arg("sysroot")
        .output()
        .expect("rustc not found")
        .stdout;
    let sysroot = std::str::from_utf8(&sysroot).expect("sysroot is not utf8").trim();
    let sysroot = &Path::new(&sysroot);
    let host = std::process::Command::new("rustc")
        .arg("-vV")
        .output()
        .expect("rustc not found for -vV")
        .stdout;
    let host = std::str::from_utf8(&host).expect("sysroot is not utf8");
    let host = host.split("\nhost: ").skip(1).next().expect("no host: part in rustc -vV");
    let host = host.split("\n").next().expect("no \n after host");
    run_pass();
    for_all_targets(&sysroot, |target| {
        miri_pass("tests/run-pass", &target, host);
        if let Ok(path) = std::env::var("MIRI_RUSTC_TEST") {
            miri_pass(&path, &target, host);
        }
    });
    compile_fail(&sysroot);
}
