// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern crate tempdir;

use tempdir::TempDir;
use std::env;
use std::process::Command;
use std::path::Path;
use std::fs::File;
use std::io::Write;

const TEST_REPOS: &'static [(&'static str, &'static str, Option<&'static str>)] = &[
    ("https://github.com/rust-lang/cargo",
     "fae9c539388f1b7c70c31fd0a21b5dd9cd071177",
     None),
    ("https://github.com/iron/iron",
     "16c858ec2901e2992fe5e529780f59fa8ed12903",
     Some(include_str!("lockfiles/iron-Cargo.lock")))
];


fn main() {
    let ref cargo = env::args().collect::<Vec<_>>()[1];
    let ref cargo = Path::new(cargo);

    for &(repo, sha, lockfile) in TEST_REPOS.iter().rev() {
        test_repo(cargo, repo, sha, lockfile);
    }
}

fn test_repo(cargo: &Path, repo: &str, sha: &str, lockfile: Option<&str>) {
    println!("testing {}", repo);
    let dir = clone_repo(repo, sha);
    if let Some(lockfile) = lockfile {
        File::create(&dir.path().join("Cargo.lock")).expect("")
            .write_all(lockfile.as_bytes()).expect("");
    }
    if !run_cargo_test(cargo, dir.path()) {
        panic!("tests failed for {}", repo);
    }
}

fn clone_repo(repo: &str, sha: &str) -> TempDir {
    let dir = TempDir::new("cargotest").expect("");
    let status = Command::new("git")
        .arg("init")
        .arg(dir.path())
        .status()
        .expect("");
    assert!(status.success());

    // Try progressively deeper fetch depths to find the commit
    let mut found = false;
    for depth in &[1, 10, 100, 1000, 100000] {
        let status = Command::new("git")
            .arg("fetch")
            .arg(repo)
            .arg("master")
            .arg(&format!("--depth={}", depth))
            .current_dir(dir.path())
            .status()
            .expect("");
        assert!(status.success());

        let status = Command::new("git")
            .arg("reset")
            .arg(sha)
            .arg("--hard")
            .current_dir(dir.path())
            .status()
            .expect("");

        if status.success() {
            found = true;
            break;
        }
    }

    if !found { panic!("unable to find commit {}", sha) }

    dir
}

fn run_cargo_test(cargo_path: &Path, crate_path: &Path) -> bool {
    let status = Command::new(cargo_path)
        .arg("test")
        // Disable rust-lang/cargo's cross-compile tests
        .env("CFG_DISABLE_CROSS_TESTS", "1")
        .current_dir(crate_path)
        .status()
        .expect("");

    status.success()
}
