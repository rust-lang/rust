//! Runs rustfmt on the repository.

use crate::{util, Build};
use std::process::Command;

pub fn format(build: &Build, check: bool) {
    let target = &build.build;
    let rustfmt_path = build.config.initial_rustfmt.as_ref().unwrap_or_else(|| {
        eprintln!("./x.py fmt is not supported on this channel");
        std::process::exit(1);
    }).clone();
    let cargo_fmt_path = rustfmt_path.with_file_name(util::exe("cargo-fmt", &target));
    assert!(cargo_fmt_path.is_file(), "{} not a file", cargo_fmt_path.display());

    let mut cmd = Command::new(&cargo_fmt_path);
    // cargo-fmt calls rustfmt as a bare command, so we need it to only find the correct one
    cmd.env("PATH", cargo_fmt_path.parent().unwrap());
    cmd.current_dir(&build.src);
    cmd.arg("fmt");

    if check {
        cmd.arg("--");
        cmd.arg("--check");
    }

    let status = cmd.status().expect("executing cargo-fmt");
    assert!(status.success(), "cargo-fmt errored with status {:?}", status);
}
