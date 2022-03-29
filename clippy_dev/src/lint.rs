use crate::cargo_clippy_path;
use std::process::{self, Command, ExitStatus};
use std::{fs, io};

fn exit_if_err(status: io::Result<ExitStatus>) {
    match status.expect("failed to run command").code() {
        Some(0) => {},
        Some(n) => process::exit(n),
        None => {
            eprintln!("Killed by signal");
            process::exit(1);
        },
    }
}

pub fn run(path: &str) {
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
                .status(),
        );
    } else {
        exit_if_err(Command::new("cargo").arg("build").status());

        // Run in a tempdir as changes to clippy do not retrigger linting
        let target = tempfile::Builder::new()
            .prefix("clippy")
            .tempdir()
            .expect("failed to create tempdir");

        let status = Command::new(cargo_clippy_path())
            .current_dir(path)
            .env("CARGO_TARGET_DIR", target.as_ref())
            .status();

        target.close().expect("failed to remove tempdir");
        exit_if_err(status);
    }
}
