// ignore-tidy-linelength

// Check that the `CURRENT_RUSTC_VERSION` placeholder is correctly replaced by the current
// `rustc` version and the `since` property in feature stability gating is properly respected.

extern crate run_make_support;

use std::path::PathBuf;

use run_make_support::{rustc, aux_build};

fn main() {
    aux_build().input("stable.rs").emit("metadata").run();

    let mut stable_path = PathBuf::from(env!("TMPDIR"));
    stable_path.push("libstable.rmeta");

    let output = rustc()
        .input("main.rs")
        .emit("metadata")
        .extern_("stable", &stable_path)
        .output();

    let stderr = String::from_utf8_lossy(&output.stderr);
    let version = include_str!(concat!(env!("S"), "/src/version"));
    let expected_string = format!("stable since {}", version.trim());
    assert!(stderr.contains(&expected_string));
}
