//! This will build the proc macro in `imp`, and copy the resulting dylib artifact into the
//! `OUT_DIR`.
//!
//! `proc-macro-test` itself contains only a path to that artifact.
//!
//! The `PROC_MACRO_TEST_TOOLCHAIN` environment variable can be exported to use
//! a specific rustup toolchain: this allows testing against older ABIs (e.g.
//! 1.58) and future ABIs (stage1, nightly)

#![allow(clippy::disallowed_methods)]

use std::{
    env,
    path::{Path, PathBuf},
    process::Command,
};

use cargo_metadata::Message;

fn main() {
    println!("cargo:rerun-if-changed=imp");

    let cargo = env::var_os("CARGO").unwrap_or_else(|| "cargo".into());

    let has_features = env::var_os("RUSTC_BOOTSTRAP").is_some()
        || String::from_utf8(Command::new(&cargo).arg("--version").output().unwrap().stdout)
            .unwrap()
            .contains("nightly");

    let out_dir = env::var_os("OUT_DIR").unwrap();
    let out_dir = Path::new(&out_dir);

    if !has_features {
        println!("proc-macro-test testing only works on nightly toolchains");
        println!("cargo::rustc-env=PROC_MACRO_TEST_LOCATION=\"\"");
        return;
    }

    let name = "proc-macro-test-impl";
    let version = "0.0.0";

    let imp_dir = std::env::current_dir().unwrap().join("imp");

    let staging_dir = out_dir.join("proc-macro-test-imp-staging");
    // this'll error out if the staging dir didn't previously exist. using
    // `std::fs::exists` would suffer from TOCTOU so just do our best to
    // wipe it and ignore errors.
    let _ = std::fs::remove_dir_all(&staging_dir);

    println!("Creating {}", staging_dir.display());
    std::fs::create_dir_all(&staging_dir).unwrap();

    let src_dir = staging_dir.join("src");
    println!("Creating {}", src_dir.display());
    std::fs::create_dir_all(src_dir).unwrap();

    for item_els in [&["Cargo.toml"][..], &["build.rs"][..], &["src", "lib.rs"]] {
        let mut src = imp_dir.clone();
        let mut dst = staging_dir.clone();
        for el in item_els {
            src.push(el);
            dst.push(el);
        }
        println!("Copying {} to {}", src.display(), dst.display());
        std::fs::copy(src, dst).unwrap();
    }

    let target_dir = out_dir.join("target");

    let mut cmd = Command::new(&cargo);
    cmd.current_dir(&staging_dir)
        .args(["build", "-p", "proc-macro-test-impl", "--message-format", "json"])
        // Explicit override the target directory to avoid using the same one which the parent
        // cargo is using, or we'll deadlock.
        // This can happen when `CARGO_TARGET_DIR` is set or global config forces all cargo
        // instance to use the same target directory.
        .arg("--target-dir")
        .arg(&target_dir);

    if let Ok(target) = std::env::var("TARGET") {
        cmd.args(["--target", &target]);
    }

    println!("Running {cmd:?}");

    let output = cmd.output().unwrap();
    if !output.status.success() {
        println!("proc-macro-test-impl failed to build");
        println!("============ stdout ============");
        println!("{}", String::from_utf8_lossy(&output.stdout));
        println!("============ stderr ============");
        println!("{}", String::from_utf8_lossy(&output.stderr));
        panic!("proc-macro-test-impl failed to build");
    }

    // Old Package ID Spec
    let repr = format!("{name} {version}");
    // New Package Id Spec since rust-lang/cargo#13311
    let pkgid = String::from_utf8(
        Command::new(cargo)
            .current_dir(&staging_dir)
            .args(["pkgid", name])
            .output()
            .unwrap()
            .stdout,
    )
    .unwrap();
    let pkgid = pkgid.trim();

    let mut artifact_path = None;
    for message in Message::parse_stream(output.stdout.as_slice()) {
        if let Message::CompilerArtifact(artifact) = message.unwrap()
            && artifact.target.kind.contains(&cargo_metadata::TargetKind::ProcMacro)
            && (artifact.package_id.repr.starts_with(&repr) || artifact.package_id.repr == pkgid)
        {
            artifact_path = Some(PathBuf::from(&artifact.filenames[0]));
        }
    }

    // This file is under `target_dir` and is already under `OUT_DIR`.
    let artifact_path = artifact_path.expect("no dylib for proc-macro-test-impl found");

    println!("cargo::rustc-env=PROC_MACRO_TEST_LOCATION={}", artifact_path.display());
}
