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

const TEST_REPOS: &'static [Test] = &[
    Test {
        name: "cargo",
        repo: "https://github.com/rust-lang/cargo",
        sha: "2324c2bbaf7fc6ea9cbdd77c034ef1af769cb617",
        lock: None,
    },
    Test {
        name: "iron",
        repo: "https://github.com/iron/iron",
        sha: "16c858ec2901e2992fe5e529780f59fa8ed12903",
        lock: Some(include_str!("lockfiles/iron-Cargo.lock")),
    },
    Test {
        name: "ripgrep",
        repo: "https://github.com/BurntSushi/ripgrep",
        sha: "b65bb37b14655e1a89c7cd19c8b011ef3e312791",
        lock: None,
    },
    Test {
        name: "tokei",
        repo: "https://github.com/Aaronepower/tokei",
        sha: "5e11c4852fe4aa086b0e4fe5885822fbe57ba928",
        lock: None,
    },
    Test {
        name: "treeify",
        repo: "https://github.com/dzamlo/treeify",
        sha: "999001b223152441198f117a68fb81f57bc086dd",
        lock: None,
    },
    Test {
        name: "xsv",
        repo: "https://github.com/BurntSushi/xsv",
        sha: "a9a7163f2a2953cea426fee1216bec914fe2f56a",
        lock: None,
    },
];

fn main() {
    // One of the projects being tested here is Cargo, and when being tested
    // Cargo will at some point call `nmake.exe` on Windows MSVC. Unfortunately
    // `nmake` will read these two environment variables below and try to
    // intepret them. We're likely being run, however, from MSYS `make` which
    // uses the same variables.
    //
    // As a result, to prevent confusion and errors, we remove these variables
    // from our environment to prevent passing MSYS make flags to nmake, causing
    // it to blow up.
    if cfg!(target_env = "msvc") {
        env::remove_var("MAKE");
        env::remove_var("MAKEFLAGS");
    }

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
