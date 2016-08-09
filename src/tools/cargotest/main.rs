// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::env;
use std::process::Command;
use std::path::{Path, PathBuf};
use std::fs::File;
use std::io::Write;

struct Test {
    repo: &'static str,
    name: &'static str,
    sha: &'static str,
    lock: Option<&'static str>,
}

const TEST_REPOS: &'static [Test] = &[Test {
                                          name: "cargo",
                                          repo: "https://github.com/rust-lang/cargo",
                                          sha: "2d85908217f99a30aa5f68e05a8980704bb71fad",
                                          lock: None,
                                      },
                                      Test {
                                          name: "iron",
                                          repo: "https://github.com/iron/iron",
                                          sha: "16c858ec2901e2992fe5e529780f59fa8ed12903",
                                          lock: Some(include_str!("lockfiles/iron-Cargo.lock")),
                                      }];


fn main() {
    let args = env::args().collect::<Vec<_>>();
    let ref cargo = args[1];
    let out_dir = Path::new(&args[2]);
    let ref cargo = Path::new(cargo);

    for test in TEST_REPOS.iter().rev() {
        test_repo(cargo, out_dir, test);
    }
}

fn test_repo(cargo: &Path, out_dir: &Path, test: &Test) {
    println!("testing {}", test.repo);
    let dir = clone_repo(test, out_dir);
    if let Some(lockfile) = test.lock {
        File::create(&dir.join("Cargo.lock"))
            .expect("")
            .write_all(lockfile.as_bytes())
            .expect("");
    }
    if !run_cargo_test(cargo, &dir) {
        panic!("tests failed for {}", test.repo);
    }
}

fn clone_repo(test: &Test, out_dir: &Path) -> PathBuf {
    let out_dir = out_dir.join(test.name);

    if !out_dir.join(".git").is_dir() {
        let status = Command::new("git")
                         .arg("init")
                         .arg(&out_dir)
                         .status()
                         .expect("");
        assert!(status.success());
    }

    // Try progressively deeper fetch depths to find the commit
    let mut found = false;
    for depth in &[0, 1, 10, 100, 1000, 100000] {
        if *depth > 0 {
            let status = Command::new("git")
                             .arg("fetch")
                             .arg(test.repo)
                             .arg("master")
                             .arg(&format!("--depth={}", depth))
                             .current_dir(&out_dir)
                             .status()
                             .expect("");
            assert!(status.success());
        }

        let status = Command::new("git")
                         .arg("reset")
                         .arg(test.sha)
                         .arg("--hard")
                         .current_dir(&out_dir)
                         .status()
                         .expect("");

        if status.success() {
            found = true;
            break;
        }
    }

    if !found {
        panic!("unable to find commit {}", test.sha)
    }
    let status = Command::new("git")
                     .arg("clean")
                     .arg("-fdx")
                     .current_dir(&out_dir)
                     .status()
                     .unwrap();
    assert!(status.success());

    out_dir
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
