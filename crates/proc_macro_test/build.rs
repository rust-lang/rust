//! This will build the proc macro in `imp`, and copy the resulting dylib artifact into the
//! `OUT_DIR`.
//!
//! `proc_macro_test` itself contains only a path to that artifact.

use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

use cargo_metadata::Message;

fn main() {
    let out_dir = env::var_os("OUT_DIR").unwrap();
    let out_dir = Path::new(&out_dir);

    let name = "proc_macro_test_impl";
    let version = "0.0.0";
    let target_dir = out_dir.join("target");
    let output = Command::new(toolchain::cargo())
        .current_dir("imp")
        .args(&["build", "-p", "proc_macro_test_impl", "--message-format", "json"])
        // Explicit override the target directory to avoid using the same one which the parent
        // cargo is using, or we'll deadlock.
        // This can happen when `CARGO_TARGET_DIR` is set or global config forces all cargo
        // instance to use the same target directory.
        .arg("--target-dir")
        .arg(&target_dir)
        .output()
        .unwrap();
    assert!(output.status.success());

    let mut artifact_path = None;
    for message in Message::parse_stream(output.stdout.as_slice()) {
        match message.unwrap() {
            Message::CompilerArtifact(artifact) => {
                if artifact.target.kind.contains(&"proc-macro".to_string()) {
                    let repr = format!("{} {}", name, version);
                    if artifact.package_id.repr.starts_with(&repr) {
                        artifact_path = Some(PathBuf::from(&artifact.filenames[0]));
                    }
                }
            }
            _ => (), // Unknown message
        }
    }

    // This file is under `target_dir` and is already under `OUT_DIR`.
    let artifact_path = artifact_path.expect("no dylib for proc_macro_test_impl found");

    let info_path = out_dir.join("proc_macro_test_location.txt");
    fs::write(info_path, artifact_path.to_str().unwrap()).unwrap();
}
