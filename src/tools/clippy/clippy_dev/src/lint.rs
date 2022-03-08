use std::process::{self, Command};

pub fn run(filename: &str) {
    let code = Command::new("cargo")
        .args(["run", "--bin", "clippy-driver", "--"])
        .args(["-L", "./target/debug"])
        .args(["-Z", "no-codegen"])
        .args(["--edition", "2021"])
        .arg(filename)
        .status()
        .expect("failed to run cargo")
        .code();

    if code.is_none() {
        eprintln!("Killed by signal");
    }

    process::exit(code.unwrap_or(1));
}
