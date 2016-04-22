// error-pattern:no mir for DefId

use std::env;
use std::process::{Command, Output};

fn run_miri(file: &str, sysroot: &str) -> Output {
    let path = env::current_dir().unwrap();
    let libpath = path.join("target").join("debug");
    let libpath = libpath.to_str().unwrap();
    let libpath2 = path.join("target").join("debug").join("deps");
    let libpath2 = libpath2.to_str().unwrap();
    Command::new("cargo")
        .args(&[
            "run", "--",
            "--sysroot", sysroot,
            "-L", libpath,
            "-L", libpath2,
            file
        ])
        .output()
        .unwrap_or_else(|e| panic!("failed to execute process: {}", e))
}

fn main() {
    let sysroot = env::var("RUST_SYSROOT").expect("env variable `RUST_SYSROOT` not set");
    let test_run = run_miri("src/bin/miri.rs", &sysroot);

    if test_run.status.code().unwrap_or(-1) != 0 {
        println!("{}", String::from_utf8(test_run.stdout).unwrap());
        panic!("{}", String::from_utf8(test_run.stderr).unwrap());
    }
}
