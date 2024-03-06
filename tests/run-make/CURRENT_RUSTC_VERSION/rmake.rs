// ignore-tidy-linelength

extern crate run_make_support;

use std::path::PathBuf;

use run_make_support::{aux_build, rustc};

fn main() {
    aux_build()
        .arg("--emit=metadata")
        .arg("stable.rs")
        .run();
    let mut stable_path = PathBuf::from(env!("TMPDIR"));
    stable_path.push("libstable.rmeta");
    let output = rustc()
        .arg("--emit=metadata")
        .arg("--extern")
        .arg(&format!("stable={}", &stable_path.to_string_lossy()))
        .arg("main.rs")
        .run();

    let stderr = String::from_utf8_lossy(&output.stderr);
    let version = include_str!(concat!(env!("S"), "/src/version"));
    let expected_string = format!("stable since {}", version.trim());
    assert!(stderr.contains(&expected_string));
}
