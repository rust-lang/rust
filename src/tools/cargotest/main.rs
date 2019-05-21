#![deny(rust_2018_idioms)]

use std::env;
use std::process::Command;
use std::path::{Path, PathBuf};
use std::fs;

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
        sha: "ad9befbc1d3b5c695e7f6b6734ee1b8e683edd41",
        lock: None,
        packages: &[],
    },
    Test {
        name: "tokei",
        repo: "https://github.com/XAMPPRocky/tokei",
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
        sha: "987e376ca7a4245dbc3e0c06e963278ee1ac92d1",
        lock: None,
        // Only test Stylo a.k.a. Quantum CSS, the parts of Servo going into Firefox.
        // This takes much less time to build than all of Servo and supports stable Rust.
        packages: &["selectors"],
    },
    Test {
        name: "webrender",
        repo: "https://github.com/servo/webrender",
        sha: "cdadd068f4c7218bd983d856981d561e605270ab",
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
        fs::write(&dir.join("Cargo.lock"), lockfile).unwrap();
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
