// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::path::PathBuf;
use std::process::Command;

fn rustc_sysroot_path() -> PathBuf {
    option_env!("SYSROOT")
        .map(String::from)
        .or_else(|| std::env::var("SYSROOT").ok())
        .or_else(|| {
            let home = option_env!("RUSTUP_HOME").or(option_env!("MULTIRUST_HOME"));
            let toolchain = option_env!("RUSTUP_TOOLCHAIN").or(option_env!("MULTIRUST_TOOLCHAIN"));
            home.and_then(|home| toolchain.map(|toolchain| format!("{}/toolchains/{}", home, toolchain)))
        })
        .or_else(|| {
            Command::new("rustc")
                .arg("--print")
                .arg("sysroot")
                .output()
                .ok()
                .and_then(|out| String::from_utf8(out.stdout).ok())
                .map(|s| s.trim().to_owned())
        })
        .expect("need to specify SYSROOT env var during clippy compilation, or use rustup or multirust")
        .into()
}

#[test]
fn dogfood() {
    if option_env!("RUSTC_TEST_SUITE").is_some() || cfg!(windows) {
        return;
    }
    let root_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let clippy_cmd = std::path::Path::new(&root_dir)
        .join("target")
        .join(env!("PROFILE"))
        .join("cargo-clippy");

    let output = std::process::Command::new(clippy_cmd)
        .current_dir(root_dir)
        .env("CLIPPY_DOGFOOD", "1")
        .env("RUSTFLAGS", format!("--sysroot {}", rustc_sysroot_path().display()))
        .arg("clippy")
        .arg("--all-targets")
        .arg("--all-features")
        .arg("--")
        .args(&["-D", "clippy::all"])
        .args(&["-D", "clippy::internal"])
        .args(&["-D", "clippy::pedantic"])
        .output()
        .unwrap();
    println!("status: {}", output.status);
    println!("stdout: {}", String::from_utf8_lossy(&output.stdout));
    println!("stderr: {}", String::from_utf8_lossy(&output.stderr));

    assert!(output.status.success());
}

#[test]
fn dogfood_tests() {
    if option_env!("RUSTC_TEST_SUITE").is_some() || cfg!(windows) {
        return;
    }
    let root_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let clippy_cmd = std::path::Path::new(&root_dir)
        .join("target")
        .join(env!("PROFILE"))
        .join("cargo-clippy");

    for d in &[
        "clippy_workspace_tests",
        "clippy_workspace_tests/src",
        "clippy_workspace_tests/subcrate",
        "clippy_workspace_tests/subcrate/src",
        "clippy_dev",
        "rustc_tools_util",
    ] {
        let output = std::process::Command::new(&clippy_cmd)
            .current_dir(root_dir.join(d))
            .env("CLIPPY_DOGFOOD", "1")
            .env("RUSTFLAGS", format!("--sysroot {}", rustc_sysroot_path().display()))
            .arg("clippy")
            .arg("--")
            .args(&["-D", "clippy::all"])
            .args(&["-D", "clippy::pedantic"])
            .output()
            .unwrap();
        println!("status: {}", output.status);
        println!("stdout: {}", String::from_utf8_lossy(&output.stdout));
        println!("stderr: {}", String::from_utf8_lossy(&output.stderr));

        assert!(output.status.success());
    }
}
