use std::path::PathBuf;
use std::process::Command;

use serde_derive::Deserialize;

use crate::cache::INTERNER;
use crate::util::output;
use crate::{Build, Crate};

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
            let name = INTERNER.intern_string(package.name);
            let mut path = PathBuf::from(package.manifest_path);
            path.pop();
            let deps = package
                .dependencies
                .into_iter()
                .filter(|dep| dep.source.is_none())
                .map(|dep| INTERNER.intern_string(dep.name))
                .collect();
            let krate = Crate { name, deps, path };
            let relative_path = krate.local_path(build);
            build.crates.insert(name, krate);
            let existing_path = build.crate_paths.insert(relative_path, name);
            assert!(existing_path.is_none(), "multiple crates with the same path");
        }
    }
}

/// Invokes `cargo metadata` to get package metadata of each workspace member.
///
/// Note that `src/tools/cargo` is no longer a workspace member but we still
/// treat it as one here, by invoking an additional `cargo metadata` command.
fn workspace_members(build: &Build) -> impl Iterator<Item = Package> {
    let cmd_metadata = |manifest_path| {
        let mut cargo = Command::new(&build.initial_cargo);
        cargo
            .arg("metadata")
            .arg("--format-version")
            .arg("1")
            .arg("--no-deps")
            .arg("--manifest-path")
            .arg(manifest_path);
        cargo
    };

    // Collects `metadata.packages` from the root workspace.
    let root_manifest_path = build.src.join("Cargo.toml");
    let root_output = output(&mut cmd_metadata(&root_manifest_path));
    let Output { packages, .. } = serde_json::from_str(&root_output).unwrap();

    // Collects `metadata.packages` from src/tools/cargo separately.
    let cargo_manifest_path = build.src.join("src/tools/cargo/Cargo.toml");
    let cargo_output = output(&mut cmd_metadata(&cargo_manifest_path));
    let Output { packages: cargo_packages, .. } = serde_json::from_str(&cargo_output).unwrap();

    // We only care about the root package from `src/tool/cargo` workspace.
    let cargo_package = cargo_packages.into_iter().find(|pkg| pkg.name == "cargo").into_iter();
    packages.into_iter().chain(cargo_package)
}
