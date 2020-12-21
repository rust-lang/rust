#![allow(clippy::filter_map)]

use std::fs;
use std::fs::File;
use std::io::prelude::*;
use std::path::PathBuf;

// This module takes an absolute path to a rustc repo and alters the dependencies to point towards
// the respective rustc subcrates instead of using extern crate xyz.
// This allows rust analyzer to analyze rustc internals and show proper information inside clippy
// code. See https://github.com/rust-analyzer/rust-analyzer/issues/3517 and https://github.com/rust-lang/rust-clippy/issues/5514 for details

pub fn run(rustc_path: Option<&str>) {
    // we can unwrap here because the arg is required by clap
    let rustc_path = PathBuf::from(rustc_path.unwrap());
    assert!(rustc_path.is_dir(), "path is not a directory");
    let rustc_source_basedir = rustc_path.join("compiler");
    assert!(
        rustc_source_basedir.is_dir(),
        "are you sure the path leads to a rustc repo?"
    );

    let clippy_root_manifest = fs::read_to_string("Cargo.toml").expect("failed to read ./Cargo.toml");
    let clippy_root_lib_rs = fs::read_to_string("src/driver.rs").expect("failed to read ./src/driver.rs");
    inject_deps_into_manifest(
        &rustc_source_basedir,
        "Cargo.toml",
        &clippy_root_manifest,
        &clippy_root_lib_rs,
    )
    .expect("Failed to inject deps into ./Cargo.toml");

    let clippy_lints_manifest =
        fs::read_to_string("clippy_lints/Cargo.toml").expect("failed to read ./clippy_lints/Cargo.toml");
    let clippy_lints_lib_rs =
        fs::read_to_string("clippy_lints/src/lib.rs").expect("failed to read ./clippy_lints/src/lib.rs");
    inject_deps_into_manifest(
        &rustc_source_basedir,
        "clippy_lints/Cargo.toml",
        &clippy_lints_manifest,
        &clippy_lints_lib_rs,
    )
    .expect("Failed to inject deps into ./clippy_lints/Cargo.toml");
}

fn inject_deps_into_manifest(
    rustc_source_dir: &PathBuf,
    manifest_path: &str,
    cargo_toml: &str,
    lib_rs: &str,
) -> std::io::Result<()> {
    // do not inject deps if we have aleady done so
    if cargo_toml.contains("[target.'cfg(NOT_A_PLATFORM)'.dependencies]") {
        eprintln!(
            "cargo dev ra_setup: warning: deps already found inside {}, doing nothing.",
            manifest_path
        );
        return Ok(());
    }

    let extern_crates = lib_rs
        .lines()
        // get the deps
        .filter(|line| line.starts_with("extern crate"))
        // we have something like "extern crate foo;", we only care about the "foo"
        //              ↓          ↓
        // extern crate rustc_middle;
        .map(|s| &s[13..(s.len() - 1)]);

    let new_deps = extern_crates.map(|dep| {
        // format the dependencies that are going to be put inside the Cargo.toml
        format!(
            "{dep} = {{ path = \"{source_path}/{dep}\" }}\n",
            dep = dep,
            source_path = rustc_source_dir.display()
        )
    });

    // format a new [dependencies]-block with the new deps we need to inject
    let mut all_deps = String::from("[target.'cfg(NOT_A_PLATFORM)'.dependencies]\n");
    new_deps.for_each(|dep_line| {
        all_deps.push_str(&dep_line);
    });
    all_deps.push_str("\n[dependencies]\n");

    // replace "[dependencies]" with
    // [dependencies]
    // dep1 = { path = ... }
    // dep2 = { path = ... }
    // etc
    let new_manifest = cargo_toml.replacen("[dependencies]\n", &all_deps, 1);

    // println!("{}", new_manifest);
    let mut file = File::create(manifest_path)?;
    file.write_all(new_manifest.as_bytes())?;

    println!("Dependency paths injected: {}", manifest_path);

    Ok(())
}
