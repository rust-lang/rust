// error-pattern:no mir for `env_logger/

use std::env;
use std::process::{Command, Output};

fn run_miri(file: &str, sysroot: &str) -> Output {
    let path = env::current_dir().unwrap();
    let libpath = path.join("target").join("debug");
    let libpath = libpath.to_str().unwrap();
    let libpath2 = path.join("target").join("debug").join("deps");
    let libpath2 = libpath2.to_str().unwrap();
    let mut args = vec![
        "run".to_string(), "--".to_string(),
        "--sysroot".to_string(), sysroot.to_string(),
        "-L".to_string(), libpath.to_string(),
        "-L".to_string(), libpath2.to_string(),
        file.to_string()
    ];
    for file in std::fs::read_dir("target/debug/deps").unwrap() {
        let file = file.unwrap();
        if file.file_type().unwrap().is_file() {
            let path = file.path();
            if let Some(ext) = path.extension() {
                if ext == "rlib" {
                    let name = path.file_stem().unwrap().to_str().unwrap();
                    if let Some(dash) = name.rfind('-') {
                        if name.starts_with("lib") {
                            args.push("--extern".to_string());
                            args.push(format!("{}={}", &name[3..dash], path.to_str().unwrap()));
                        }
                    }
                }
            }
        }
    }
    Command::new("cargo")
        .args(&args)
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
