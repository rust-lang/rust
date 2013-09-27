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
pub enum WhatToBuild {
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
        Some(s) if s == stem => true,
        _ => false
    }
}

pub fn lib_name_of(p: &Path) -> Path {
    p.push("lib.rs")
}

pub static lib_crate_filename: &'static str = "lib.rs";
