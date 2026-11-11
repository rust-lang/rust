//! This module interacts with Cargo metadata to collect and store information about
//! the packages in the Rust workspace.
//!
//! It runs `cargo metadata` to gather details about each package, including its name,
//! source, dependencies, targets, and available features. The collected metadata is then
//! used to update the `Build` structure, ensuring proper dependency resolution and
//! compilation flow.

use std::collections::{BTreeMap, HashSet};
use std::path::PathBuf;

use serde_derive::Deserialize;

use crate::core::session::Session;
use crate::utils::exec::command;
use crate::utils::helpers::t;

#[derive(Debug, Clone)]
pub(crate) struct Crate {
    pub(crate) name: String,
    pub(crate) deps: HashSet<String>,
    pub(crate) path: PathBuf,
    pub(crate) features: Vec<String>,
}

impl Crate {
    pub(crate) fn local_path(&self, sess: &Session) -> PathBuf {
        self.path.strip_prefix(&sess.config.src).unwrap().into()
    }
}

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

/// Collects and stores package metadata of each workspace members into `sess`,
/// by executing `cargo metadata` commands.
pub(crate) fn build(sess: &mut Session) {
    for package in workspace_members(sess) {
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
            let relative_path = krate.local_path(sess);
            sess.crates.insert(name.clone(), krate);
            let existing_path = sess.crate_paths.insert(relative_path, name);
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
fn workspace_members(sess: &Session) -> Vec<Package> {
    let collect_metadata = |manifest_path| {
        let mut cargo = command(&sess.initial_cargo);
        cargo
            // Will read the libstd Cargo.toml
            // which uses the unstable `public-dependency` feature.
            .env("RUSTC_BOOTSTRAP", "1")
            .arg("metadata")
            .arg("--format-version")
            .arg("1")
            .arg("--no-deps")
            .arg("--manifest-path")
            .arg(sess.src.join(manifest_path));
        let metadata_output = cargo.run_in_dry_run().run_capture_stdout(sess).stdout();
        let Output { packages, .. } = t!(serde_json::from_str(&metadata_output));
        packages
    };

    // Collects `metadata.packages` from the root and library workspaces.
    let mut packages = vec![];
    packages.extend(collect_metadata("Cargo.toml"));
    packages.extend(collect_metadata("library/Cargo.toml"));
    packages
}
