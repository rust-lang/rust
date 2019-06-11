use std::collections::HashMap;
use std::process::Command;
use std::path::PathBuf;
use std::collections::HashSet;

use build_helper::output;
use serde::Deserialize;
use serde_json;

use crate::{Build, Crate};
use crate::cache::INTERNER;

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
    let mut resolves = Vec::new();
    build_krate(&build.std_features(), build, &mut resolves, "src/libstd");
    build_krate("", build, &mut resolves, "src/libtest");
    build_krate(&build.rustc_features(), build, &mut resolves, "src/rustc");

    let mut id2name = HashMap::with_capacity(build.crates.len());
    for (name, krate) in build.crates.iter() {
        id2name.insert(krate.id.clone(), name.clone());
    }

    for node in resolves {
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

fn build_krate(features: &str, build: &mut Build, resolves: &mut Vec<ResolveNode>, krate: &str) {
    // Run `cargo metadata` to figure out what crates we're testing.
    //
    // Down below we're going to call `cargo test`, but to test the right set
    // of packages we're going to have to know what `-p` arguments to pass it
    // to know what crates to test. Here we run `cargo metadata` to learn about
    // the dependency graph and what `-p` arguments there are.
    let mut cargo = Command::new(&build.initial_cargo);
    cargo.arg("metadata")
         .arg("--format-version").arg("1")
         .arg("--features").arg(features)
         .arg("--manifest-path").arg(build.src.join(krate).join("Cargo.toml"));
    let output = output(&mut cargo);
    let output: Output = serde_json::from_str(&output).unwrap();
    for package in output.packages {
        if package.source.is_none() {
            let name = INTERNER.intern_string(package.name);
            let mut path = PathBuf::from(package.manifest_path);
            path.pop();
            build.crates.insert(name, Crate {
                name,
                id: package.id,
                deps: HashSet::new(),
                path,
            });
        }
    }
    resolves.extend(output.resolve.nodes);
}
