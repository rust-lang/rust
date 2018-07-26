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
    packages: &'static [&'static str],
}

const TEST_REPOS: &'static [Test] = &[
    Test {
        name: "iron",
        repo: "https://github.com/iron/iron",
        sha: "21c7dae29c3c214c08533c2a55ac649b418f2fe3",
        lock: Some(include_str!("lockfiles/iron-Cargo.lock")),
        packages: &[],
    },
    Test {
        name: "ripgrep",
        repo: "https://github.com/BurntSushi/ripgrep",
        sha: "b65bb37b14655e1a89c7cd19c8b011ef3e312791",
        lock: None,
        packages: &[],
    },
    Test {
        name: "tokei",
        repo: "https://github.com/Aaronepower/tokei",
        sha: "5e11c4852fe4aa086b0e4fe5885822fbe57ba928",
        lock: None,
        packages: &[],
    },
    Test {
        name: "treeify",
        repo: "https://github.com/dzamlo/treeify",
        sha: "999001b223152441198f117a68fb81f57bc086dd",
        lock: None,
        packages: &[],
    },
    Test {
        name: "xsv",
        repo: "https://github.com/BurntSushi/xsv",
        sha: "66956b6bfd62d6ac767a6b6499c982eae20a2c9f",
        lock: None,
        packages: &[],
    },
    Test {
        name: "servo",
        repo: "https://github.com/servo/servo",
        sha: "17e97b9320fdb7cdb33bbc5f4d0fde0653bbf2e4",
        lock: None,
        // Only test Stylo a.k.a. Quantum CSS, the parts of Servo going into Firefox.
        // This takes much less time to build than all of Servo and supports stable Rust.
        packages: &["stylo_tests", "selectors"],
    },
    Test {
        name: "webrender",
        repo: "https://github.com/servo/webrender",
        sha: "57250b2b8fa63934f80e5376a29f7dcb3f759ad6",
        lock: None,
        packages: &[],
    },
];

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
    if !run_cargo_test(cargo, &dir, test.packages) {
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

fn run_cargo_test(cargo_path: &Path, crate_path: &Path, packages: &[&str]) -> bool {
    let mut command = Command::new(cargo_path);
    command.arg("test");
    for name in packages {
        command.arg("-p").arg(name);
    }
    let status = command
        // Disable rust-lang/cargo's cross-compile tests
        .env("CFG_DISABLE_CROSS_TESTS", "1")
        // Relax #![deny(warnings)] in some crates
        .env("RUSTFLAGS", "--cap-lints warn")
        .current_dir(crate_path)
        .status()
        .expect("");

    status.success()
}
