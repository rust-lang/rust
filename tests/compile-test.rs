use std::{env, fs};
use std::process::{Command, Output};

fn run_miri(file: &str, sysroot: &str) -> Output {
    Command::new("cargo")
        .args(&["run", "--", "--sysroot", sysroot, file])
        .output()
        .unwrap_or_else(|e| panic!("failed to execute process: {}", e))
}

#[test]
fn run_pass() {
    let sysroot = env::var("RUST_SYSROOT").expect("env variable `RUST_SYSROOT` not set");

    let test_files = fs::read_dir("./tests/run-pass/")
                         .expect("Can't read `run-pass` directory")
                         .filter_map(|entry| entry.ok())
                         .filter(|entry| {
                             entry.clone()
                                  .file_type()
                                  .map(|x| x.is_file())
                                  .unwrap_or(false)
                         })
                         .filter_map(|entry| entry.path().to_str().map(|x| x.to_string()));

    for file in test_files {
        println!("{}: compile test running", file);  

        let test_run = run_miri(&file, &sysroot);

        if test_run.status.code().unwrap_or(-1) != 0 {
            println!("{}: error {:?}", file, test_run);
        } else {
            println!("{}: ok", file);
        }
    }
}
