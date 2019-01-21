//! Tidy check to ensure that `[dependencies]` and `extern crate` are in sync.
//!
//! This tidy check ensures that all crates listed in the `[dependencies]`
//! section of a `Cargo.toml` are present in the corresponding `lib.rs` as
//! `extern crate` declarations. This should help us keep the DAG correctly
//! structured through various refactorings to prune out unnecessary edges.

use std::fs;
use std::path::Path;

pub fn check(path: &Path, bad: &mut bool) {
    if !super::filter_dirs(path) {
        return
    }
    for entry in t!(path.read_dir(), path).map(|e| t!(e)) {
        // Look for `Cargo.toml` with a sibling `src/lib.rs` or `lib.rs`.
        if entry.file_name().to_str() == Some("Cargo.toml") {
            if path.join("src/lib.rs").is_file() {
                verify(&entry.path(), &path.join("src/lib.rs"), bad)
            }
            if path.join("lib.rs").is_file() {
                verify(&entry.path(), &path.join("lib.rs"), bad)
            }
        } else if t!(entry.file_type()).is_dir() {
            check(&entry.path(), bad);
        }
    }
}

/// Verifies that the dependencies in Cargo.toml at `tomlfile` are synced with
/// the `extern crate` annotations in the lib.rs at `libfile`.
fn verify(tomlfile: &Path, libfile: &Path, bad: &mut bool) {
    let toml = t!(fs::read_to_string(&tomlfile));
    let librs = t!(fs::read_to_string(&libfile));

    if toml.contains("name = \"bootstrap\"") {
        return
    }

    // "Poor man's TOML parser" -- just assume we use one syntax for now.
    //
    // We just look for:
    //
    // ````
    // [dependencies]
    // name = ...
    // name2 = ...
    // name3 = ...
    // ```
    //
    // If we encounter a line starting with `[` then we assume it's the end of
    // the dependency section and bail out.
    let deps = match toml.find("[dependencies]") {
        Some(i) => &toml[i+1..],
        None => return,
    };
    for line in deps.lines() {
        if line.starts_with('[') {
            break
        }

        let mut parts = line.splitn(2, '=');
        let krate = parts.next().unwrap().trim();
        if parts.next().is_none() {
            continue
        }

        // Don't worry about depending on core/std while not writing `extern crate
        // core/std` -- that's intentional.
        if krate == "core" || krate == "std" {
            continue
        }

        // This is intentional -- this dependency just makes the crate available
        // for others later on.
        let whitelisted = krate.starts_with("panic");
        if toml.contains("name = \"std\"") && whitelisted {
            continue
        }

        if !librs.contains(&format!("extern crate {}", krate)) {
            tidy_error!(bad, "{} doesn't have `extern crate {}`, but Cargo.toml \
                              depends on it", libfile.display(), krate);
        }
    }
}
