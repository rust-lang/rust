//! This module interacts with Cargo metadata to collect and store information about
//! the packages in the Rust workspace.
//!
//! It runs `cargo metadata` to gather details about each package, including its name,
//! source, dependencies, targets, and available features. The collected metadata is then
//! used to update the `Build` structure, ensuring proper dependency resolution and
//! compilation flow.
use std::collections::BTreeMap;
use std::path::PathBuf;

use serde_derive::Deserialize;

use crate::utils::exec::command;
use crate::{Build, Crate, t};

/// For more information, see the output of
/// <https://doc.rust-lang.org/nightly/cargo/commands/cargo-metadata.html>
#[derive(Debug, Deserialize)]
struct Output {
    packages: Vec<Package>,
}

/// For more information, see the output of
/// <https://doc.rust-lang.org/nightly/cargo/commands/cargo-metadata.html>
#[derive(Debug, Deserialize)]
struct Package {
    name: String,
    source: Option<String>,
    manifest_path: String,
    dependencies: Vec<Dependency>,
    features: BTreeMap<String, Vec<String>>,
}

/// For more information, see the output of
/// <https://doc.rust-lang.org/nightly/cargo/commands/cargo-metadata.html>
#[derive(Debug, Deserialize)]
struct Dependency {
    name: String,
    source: Option<String>,
}

/// Collects and stores package metadata of each workspace members into `build`,
/// by executing `cargo metadata` commands.
pub fn build(build: &mut Build) {
    for package in workspace_members(build) {
        if package.source.is_none() {
            let name = package.name;
            let mut path = PathBuf::from(package.manifest_path);
            path.pop();
            let deps = package
                .dependencies
                .into_iter()
                .filter(|dep| dep.source.is_none())
                .map(|dep| dep.name)
                .collect();
            let krate = Crate {
                name: name.clone(),
                deps,
                path,
                features: package.features.keys().cloned().collect(),
            };
            let relative_path = krate.local_path(build);
            build.crates.insert(name.clone(), krate);
            let existing_path = build.crate_paths.insert(relative_path, name);
            assert!(
                existing_path.is_none(),
                "multiple crates with the same path: {}",
                existing_path.unwrap()
            );
        }
    }
}

/// Invokes `cargo metadata` to get package metadata of each workspace member.
///
/// This is used to resolve specific crate paths in `fn should_run` to compile
/// particular crate (e.g., `x build sysroot` to build library/sysroot).
fn workspace_members(build: &Build) -> Vec<Package> {
    let collect_metadata = |manifest_path| {
        let mut cargo = command(&build.initial_cargo);
        cargo
            // Will read the libstd Cargo.toml
            // which uses the unstable `public-dependency` feature.
            .env("RUSTC_BOOTSTRAP", "1")
            .arg("metadata")
            .arg("--format-version")
            .arg("1")
            .arg("--no-deps")
            .arg("--manifest-path")
            .arg(build.src.join(manifest_path));
        let metadata_output = cargo.run_in_dry_run().run_capture_stdout(build).stdout();
        let Output { packages, .. } = t!(serde_json::from_str(&metadata_output));
        packages
    };

    // Collects `metadata.packages` from the root and library workspaces.
    let mut packages = vec![];
    packages.extend(collect_metadata("Cargo.toml"));
    packages.extend(collect_metadata("library/Cargo.toml"));
    packages
}
