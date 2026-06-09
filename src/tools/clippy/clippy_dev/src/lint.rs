use crate::utils::{ErrAction, cargo_cmd, expect_action, run_exit_on_err};
use std::process::Command;
use std::{env, fs};

#[cfg(not(windows))]
static CARGO_CLIPPY_EXE: &str = "cargo-clippy";
#[cfg(windows)]
static CARGO_CLIPPY_EXE: &str = "cargo-clippy.exe";

pub fn run<'a>(path: &str, edition: &str, args: impl Iterator<Item = &'a String>) {
    let is_file = expect_action(fs::metadata(path), ErrAction::Read, path).is_file();
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
        // Ideally this would just be `cargo run`, but the working directory needs to be
        // set to clippy's directory when building, and the target project's directory
        // when running clippy. `cargo` can only set a single working directory for both
        // when using `run`.
        run_exit_on_err("cargo build", cargo_cmd().arg("build"));

        let mut exe = env::current_exe().expect("failed to get current executable name");
        exe.set_file_name(CARGO_CLIPPY_EXE);
        run_exit_on_err(
            "cargo clippy",
            Command::new(exe)
                .arg("clippy")
                .args(args)
                // Prevent rustc from creating `rustc-ice-*` files the console output is enough.
                .env("RUSTC_ICE", "0")
                .current_dir(path),
        );
    }
}
