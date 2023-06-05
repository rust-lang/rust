use crate::{cargo_clippy_path, exit_if_err};
use std::fs;
use std::process::{self, Command};

pub fn run<'a>(path: &str, args: impl Iterator<Item = &'a String>) {
    let is_file = match fs::metadata(path) {
        Ok(metadata) => metadata.is_file(),
        Err(e) => {
            eprintln!("Failed to read {path}: {e:?}");
            process::exit(1);
        },
    };

    if is_file {
        exit_if_err(
            Command::new("cargo")
                .args(["run", "--bin", "clippy-driver", "--"])
                .args(["-L", "./target/debug"])
                .args(["-Z", "no-codegen"])
                .args(["--edition", "2021"])
                .arg(path)
                .args(args)
                .status(),
        );
    } else {
        exit_if_err(Command::new("cargo").arg("build").status());

        let status = Command::new(cargo_clippy_path())
            .arg("clippy")
            .args(args)
            .current_dir(path)
            .status();

        exit_if_err(status);
    }
}
