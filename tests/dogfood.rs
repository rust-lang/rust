// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[test]
fn dogfood() {
    if option_env!("RUSTC_TEST_SUITE").is_some() || cfg!(windows) {
        return;
    }
    let root_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let clippy_cmd = std::path::Path::new(&root_dir).join("target/debug/cargo-clippy");

    println!("{:?}", clippy_cmd);
    let output = std::process::Command::new(clippy_cmd)
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

    for d in &[
        "clippy_workspace_tests",
        "clippy_workspace_tests/src",
        "clippy_workspace_tests/subcrate",
        "clippy_workspace_tests/subcrate/src",
        "clippy_dev",
        "rustc_tools_util",
    ] {
        let clippy_cmd = std::path::Path::new(&root_dir)
            .join("target/debug/cargo-clippy");
        std::env::set_current_dir(root_dir.join(d)).unwrap();
        let output = std::process::Command::new(clippy_cmd)
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
    std::env::set_current_dir(root_dir).unwrap();
}
