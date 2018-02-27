// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::collections::HashMap;
use std::process::Command;
use std::path::PathBuf;

use build_helper::output;
use serde_json;

use {Build, Crate};
use cache::Intern;

#[derive(Deserialize)]
struct Output {
    packages: Vec<Package>,
    resolve: Resolve,
}

#[derive(Deserialize)]
struct Package {
    id: String,
    name: String,
    version: String,
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
    build_krate(build, "src/libstd");
    build_krate(build, "src/libtest");
    build_krate(build, "src/rustc");
}

fn build_krate(build: &mut Build, krate: &str) {
    let mut cargo = Command::new(&build.config.general.initial_cargo);
    cargo
        .arg("metadata")
        .arg("--format-version")
        .arg("1")
        .arg("--manifest-path")
        .arg(build.config.src.join(krate).join("Cargo.toml"));
    let output = output(&mut cargo);
    let output: Output = serde_json::from_str(&output).unwrap();
    let mut id2name = HashMap::new();
    for package in output.packages {
        if package.source.is_none() {
            let name = package.name.intern();
            id2name.insert(package.id, name);
            let mut path = PathBuf::from(package.manifest_path);
            path.pop();
            build.crates.insert(
                name,
                Crate {
                    name,
                    version: package.version,
                    deps: Vec::new(),
                    path,
                },
            );
        }
    }

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
            krate.deps.push(*dep);
        }
    }
}
