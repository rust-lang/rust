use crate::utils::{cargo_clippy_path, cargo_cmd, run_exit_on_err};
use std::fs;
use std::process::{self, Command};

pub fn run<'a>(path: &str, edition: &str, args: impl Iterator<Item = &'a String>) {
    let is_file = match fs::metadata(path) {
        Ok(metadata) => metadata.is_file(),
        Err(e) => {
            eprintln!("Failed to read {path}: {e:?}");
            process::exit(1);
        },
    };

    if is_file {
        run_exit_on_err(
            "cargo run",
            cargo_cmd()
                .args(["run", "--bin", "clippy-driver", "--"])
                .args(["-L", "./target/debug"])
                .args(["-Z", "no-codegen"])
                .args(["--edition", edition])
                .arg(path)
                .args(args)
                // Prevent rustc from creating `rustc-ice-*` files the console output is enough.
                .env("RUSTC_ICE", "0"),
        );
    } else {
        run_exit_on_err("cargo build", cargo_cmd().arg("build"));
        run_exit_on_err(
            "cargo clippy",
            Command::new(cargo_clippy_path())
                .arg("clippy")
                .args(args)
                // Prevent rustc from creating `rustc-ice-*` files the console output is enough.
                .env("RUSTC_ICE", "0")
                .current_dir(path),
        );
    }
}
