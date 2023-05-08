use std::path::PathBuf;
use std::process::Command;

use serde_derive::Deserialize;

use crate::cache::INTERNER;
use crate::util::output;
use crate::{t, Build, Crate};

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
    targets: Vec<Target>,
}

/// For more information, see the output of
/// <https://doc.rust-lang.org/nightly/cargo/commands/cargo-metadata.html>
#[derive(Debug, Deserialize)]
struct Dependency {
    name: String,
    source: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Target {
    kind: Vec<String>,
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
            let has_lib = package.targets.iter().any(|t| t.kind.iter().any(|k| k == "lib"));
            let krate = Crate { name, deps, path, has_lib };
            let relative_path = krate.local_path(build);
            build.crates.insert(name, krate);
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
/// Note that `src/tools/cargo` is no longer a workspace member but we still
/// treat it as one here, by invoking an additional `cargo metadata` command.
fn workspace_members(build: &Build) -> impl Iterator<Item = Package> {
    let collect_metadata = |manifest_path| {
        let mut cargo = Command::new(&build.initial_cargo);
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
        let metadata_output = output(&mut cargo);
        let Output { packages, .. } = t!(serde_json::from_str(&metadata_output));
        packages
    };

    // Collects `metadata.packages` from all workspaces.
    let packages = collect_metadata("Cargo.toml");
    let cargo_packages = collect_metadata("src/tools/cargo/Cargo.toml");
    let ra_packages = collect_metadata("src/tools/rust-analyzer/Cargo.toml");
    let bootstrap_packages = collect_metadata("src/bootstrap/Cargo.toml");

    // We only care about the root package from `src/tool/cargo` workspace.
    let cargo_package = cargo_packages.into_iter().find(|pkg| pkg.name == "cargo").into_iter();

    packages.into_iter().chain(cargo_package).chain(ra_packages).chain(bootstrap_packages)
}
