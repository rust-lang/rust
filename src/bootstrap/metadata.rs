use std::collections::HashMap;
use std::collections::HashSet;
use std::path::PathBuf;
use std::process::Command;

use build_helper::output;
use serde::Deserialize;

use crate::cache::INTERNER;
use crate::{Build, Crate};

#[derive(Deserialize)]
struct Output {
    packages: Vec<Package>,
    resolve: Resolve,
}

#[derive(Deserialize)]
struct Package {
    id: String,
    name: String,
    source: Option<String>,
    manifest_path: String,
}

#[derive(Deserialize)]
struct Resolve {
    nodes: Vec<ResolveNode>,
}

#[derive(Deserialize)]
struct ResolveNode {
    id: String,
    dependencies: Vec<String>,
}

pub fn build(build: &mut Build) {
    // Run `cargo metadata` to figure out what crates we're testing.
    let features: Vec<_> = build
        .std_features()
        .split_whitespace()
        .map(|f| format!("test/{}", f))
        .chain(build.rustc_features().split_whitespace().map(|f| format!("rustc-main/{}", f)))
        .collect();
    let mut cargo = Command::new(&build.initial_cargo);
    cargo
        .arg("metadata")
        .arg("--format-version")
        .arg("1")
        .arg("--features")
        .arg(features.join(","))
        .arg("-Zpackage-features")
        .arg("--manifest-path")
        .arg(build.src.join("Cargo.toml"))
        .env("RUSTC_BOOTSTRAP", "1");
    let output = output(&mut cargo);
    let output: Output = serde_json::from_str(&output).unwrap();
    for package in output.packages {
        if package.source.is_none() {
            let name = INTERNER.intern_string(package.name);
            let mut path = PathBuf::from(package.manifest_path);
            path.pop();
            build.crates.insert(name, Crate { name, id: package.id, deps: HashSet::new(), path });
        }
    }

    let id2name: HashMap<_, _> =
        build.crates.iter().map(|(name, krate)| (krate.id.clone(), name.clone())).collect();

    for node in output.resolve.nodes {
        let name = match id2name.get(&node.id) {
            Some(name) => name,
            None => continue,
        };

        let krate = build.crates.get_mut(name).unwrap();
        for dep in node.dependencies.iter() {
            let dep = match id2name.get(dep) {
                Some(dep) => dep,
                None => continue,
            };
            krate.deps.insert(*dep);
        }
    }
}
