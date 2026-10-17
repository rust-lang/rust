#![allow(clippy::disallowed_methods, clippy::print_stdout)]
#![cfg(not(miri))]
use std::{
    env,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

fn project_root() -> PathBuf {
    PathBuf::from(
        env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| env!("CARGO_MANIFEST_DIR").to_owned()),
    )
}

fn run(cmd: &str, dir: impl AsRef<Path>) -> Result<(), ()> {
    let mut args: Vec<_> = cmd.split_whitespace().collect();
    let bin = args.remove(0);
    println!("> {}", cmd);
    let output = Command::new(bin)
        .args(args)
        .current_dir(dir)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .output()
        .map_err(drop)?;
    if output.status.success() {
        Ok(())
    } else {
        let stdout = String::from_utf8(output.stdout).map_err(drop)?;
        print!("{}", stdout);
        Err(())
    }
}

#[test]
fn check_code_formatting() {
    let dir = project_root();
    if run("rustfmt +stable --version", &dir).is_err() {
        panic!(
            "failed to run rustfmt from toolchain 'stable'; \
             please run `rustup component add rustfmt --toolchain stable` to install it.",
        );
    }
    if run("cargo +stable fmt -- --check", &dir).is_err() {
        panic!("code is not properly formatted; please format the code by running `cargo fmt`")
    }
}
