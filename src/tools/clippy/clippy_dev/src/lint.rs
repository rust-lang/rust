use crate::utils::{cargo_clippy_path, exit_if_err};
use std::process::{self, Command};
use std::{env, fs};

pub fn run<'a>(path: &str, edition: &str, args: impl Iterator<Item = &'a String>) {
    let is_file = match fs::metadata(path) {
        Ok(metadata) => metadata.is_file(),
        Err(e) => {
            eprintln!("Failed to read {path}: {e:?}");
            process::exit(1);
        },
    };

    if is_file {
        exit_if_err(
            Command::new(env::var("CARGO").unwrap_or_else(|_| "cargo".into()))
                .args(["run", "--bin", "clippy-driver", "--"])
                .args(["-L", "./target/debug"])
                .args(["-Z", "no-codegen"])
                .args(["--edition", edition])
                .arg(path)
                .args(args)
                // Prevent rustc from creating `rustc-ice-*` files the console output is enough.
                .env("RUSTC_ICE", "0")
                .status(),
        );
    } else {
        exit_if_err(
            Command::new(env::var("CARGO").unwrap_or_else(|_| "cargo".into()))
                .arg("build")
                .status(),
        );

        let status = Command::new(cargo_clippy_path())
            .arg("clippy")
            .args(args)
            // Prevent rustc from creating `rustc-ice-*` files the console output is enough.
            .env("RUSTC_ICE", "0")
            .current_dir(path)
            .status();

        exit_if_err(status);
    }
}
