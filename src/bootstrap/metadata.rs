use std::path::PathBuf;
use std::process::Command;

use serde_derive::Deserialize;

use crate::cache::INTERNER;
use crate::util::output;
use crate::{Build, Crate};

#[derive(Deserialize)]
struct Output {
    packages: Vec<Package>,
}

#[derive(Deserialize)]
struct Package {
    name: String,
    source: Option<String>,
    manifest_path: String,
    dependencies: Vec<Dependency>,
}

#[derive(Deserialize)]
struct Dependency {
    name: String,
    source: Option<String>,
}

pub fn build(build: &mut Build) {
    // Run `cargo metadata` to figure out what crates we're testing.
    let mut cargo = Command::new(&build.initial_cargo);
    cargo
        .arg("metadata")
        .arg("--format-version")
        .arg("1")
        .arg("--no-deps")
        .arg("--manifest-path")
        .arg(build.src.join("Cargo.toml"));
    let output = output(&mut cargo);
    let output: Output = serde_json::from_str(&output).unwrap();
    for package in output.packages {
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
