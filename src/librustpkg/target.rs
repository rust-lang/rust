// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


// Data types that express build artifacts

#[deriving(Eq)]
pub enum OutputType { Main, Lib, Bench, Test }

#[deriving(Eq)]
pub enum Target {
    /// In-place build
    Build,
    /// Install to bin/ or lib/ dir
    Install
}

#[deriving(Eq, Clone)]
pub struct WhatToBuild {
    build_type: BuildType, // Whether or not to ignore the pkg.rs file
    sources: SourceType, // Which crates to build
    inputs_to_discover: ~[(~str, Path)] // Inputs to these crates to be discovered
        // (For now all of these inputs will be taken as discovered inputs
        // for all of the crates)
        // (Paired with their kinds)
}

impl WhatToBuild {
    pub fn new(build_type: BuildType, sources: SourceType) -> WhatToBuild {
        WhatToBuild { build_type: build_type,
                      sources: sources,
                      inputs_to_discover: ~[] }
    }
}

#[deriving(Eq, Clone)]
pub enum BuildType {
    Inferred, // Ignore the pkg.rs file even if one exists
    MaybeCustom // Use the pkg.rs file if it exists
}

#[deriving(Eq, Clone)]
pub enum SourceType {
    /// Build just one lib.rs file in `path`, which is relative to the active workspace's src/ dir
    JustOne(Path),
    /// Build any test.rs files that can be recursively found in the active workspace
    Tests,
    /// Build everything
    Everything
}

pub fn is_lib(p: &Path) -> bool {
    file_is(p, "lib")
}

pub fn is_main(p: &Path) -> bool {
    file_is(p, "main")
}

pub fn is_test(p: &Path) -> bool {
    file_is(p, "test")
}

pub fn is_bench(p: &Path) -> bool {
    file_is(p, "bench")
}

fn file_is(p: &Path, stem: &str) -> bool {
    match p.filestem() {
        Some(s) if s == stem.as_bytes() => true,
        _ => false
    }
}

pub fn lib_name_of(p: &Path) -> Path {
    p.join("lib.rs")
}

pub static lib_crate_filename: &'static str = "lib.rs";
