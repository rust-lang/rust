//! This will build the proc macro in `imp`, and copy the resulting dylib artifact into the
//! `OUT_DIR`.
//!
//! `proc-macro-test` itself contains only a path to that artifact.

use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

use cargo_metadata::Message;

fn main() {
    let out_dir = env::var_os("OUT_DIR").unwrap();
    let out_dir = Path::new(&out_dir);

    let name = "proc-macro-test-impl";
    let version = "0.0.0";

    let imp_dir = std::env::current_dir().unwrap().join("imp");
    let staging_dir = out_dir.join("staging");
    std::fs::create_dir_all(&staging_dir).unwrap();
    std::fs::create_dir_all(staging_dir.join("src")).unwrap();

    for item_els in [&["Cargo.toml"][..], &["Cargo.lock"], &["src", "lib.rs"]] {
        let mut src = imp_dir.clone();
        let mut dst = staging_dir.clone();
        for el in item_els {
            src.push(el);
            dst.push(el);
        }
        std::fs::copy(src, dst).unwrap();
    }

    let target_dir = out_dir.join("target");
    let output = Command::new(toolchain::cargo())
        .current_dir(&staging_dir)
        .args(&["build", "-p", "proc-macro-test-impl", "--message-format", "json"])
        // Explicit override the target directory to avoid using the same one which the parent
        // cargo is using, or we'll deadlock.
        // This can happen when `CARGO_TARGET_DIR` is set or global config forces all cargo
        // instance to use the same target directory.
        .arg("--target-dir")
        .arg(&target_dir)
        .output()
        .unwrap();
    if !output.status.success() {
        println!("proc-macro-test-impl failed to build");
        println!("============ stdout ============");
        println!("{}", String::from_utf8_lossy(&output.stdout));
        println!("============ stderr ============");
        println!("{}", String::from_utf8_lossy(&output.stderr));
        panic!("proc-macro-test-impl failed to build");
    }

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
    let artifact_path = artifact_path.expect("no dylib for proc-macro-test-impl found");

    let info_path = out_dir.join("proc_macro_test_location.txt");
    fs::write(info_path, artifact_path.to_str().unwrap()).unwrap();
}
