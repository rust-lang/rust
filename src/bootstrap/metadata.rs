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
use rustc_serialize::json;

use {Build, Crate};

#[derive(RustcDecodable)]
struct Output {
    packages: Vec<Package>,
    resolve: Resolve,
}

#[derive(RustcDecodable)]
struct Package {
    id: String,
    name: String,
    source: Option<String>,
    manifest_path: String,
}

#[derive(RustcDecodable)]
struct Resolve {
    nodes: Vec<ResolveNode>,
}

#[derive(RustcDecodable)]
struct ResolveNode {
    id: String,
    dependencies: Vec<String>,
}

pub fn build(build: &mut Build) {
    build_krate(build, "src/rustc/std_shim");
    build_krate(build, "src/rustc/test_shim");
    build_krate(build, "src/rustc");
}

fn build_krate(build: &mut Build, krate: &str) {
    // Run `cargo metadata` to figure out what crates we're testing.
    //
    // Down below we're going to call `cargo test`, but to test the right set
    // of packages we're going to have to know what `-p` arguments to pass it
    // to know what crates to test. Here we run `cargo metadata` to learn about
    // the dependency graph and what `-p` arguments there are.
    let mut cargo = Command::new(&build.cargo);
    cargo.arg("metadata")
         .arg("--manifest-path").arg(build.src.join(krate).join("Cargo.toml"));
    let output = output(&mut cargo);
    let output: Output = json::decode(&output).unwrap();
    let mut id2name = HashMap::new();
    for package in output.packages {
        if package.source.is_none() {
            id2name.insert(package.id, package.name.clone());
            let mut path = PathBuf::from(package.manifest_path);
            path.pop();
            build.crates.insert(package.name.clone(), Crate {
                build_step: format!("build-crate-{}", package.name),
                doc_step: format!("doc-crate-{}", package.name),
                test_step: format!("test-crate-{}", package.name),
                bench_step: format!("bench-crate-{}", package.name),
                name: package.name,
                deps: Vec::new(),
                path: path,
            });
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
            krate.deps.push(dep.clone());
        }
    }
}
